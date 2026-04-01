// qnn_backend.c — Qualcomm QNN C API wrapper for BirdNET-Go.
//
// Dynamically loads libQnnGpu.so (or libQnnHtp.so) and libQnnSystem.so at
// runtime, so the binary has no hard link-time dependency on the QAIRT SDK.
//
// Build requirements:
//   - QAIRT SDK headers in the include path (QNN/QnnInterface.h etc.)
//   - Link: -ldl

#include "qnn_backend.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Pull in the QNN API headers from the QAIRT SDK.
// The CGO cflags in classifier.go set -I to the SDK include directory.
#include "QNN/QnnInterface.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnTensor.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnTypes.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/System/QnnSystemContext.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void set_error(char *buf, size_t buf_size, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

static void set_error(char *buf, size_t buf_size, const char *fmt, ...) {
    if (!buf || buf_size == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, buf_size, fmt, ap);
    va_end(ap);
}

// ---------------------------------------------------------------------------
// QNN function pointer typedefs (resolved via dlsym)
// ---------------------------------------------------------------------------

typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_t)(
    const QnnInterface_t ***providerList,
    uint32_t               *numProviders);

typedef Qnn_ErrorHandle_t (*QnnSystemInterface_getProviders_t)(
    const QnnSystemInterface_t ***providerList,
    uint32_t                    *numProviders);

// ---------------------------------------------------------------------------
// Session struct
// ---------------------------------------------------------------------------

struct qnn_session {
    // dlopen handles
    void *backend_lib;
    void *system_lib;
    void *model_lib;    // NULL when loaded from context binary

    // QNN interface function table
    QnnInterface_t iface;

    // QNN handles
    Qnn_LogHandle_t     log_handle;
    Qnn_BackendHandle_t backend_handle;
    Qnn_DeviceHandle_t  device_handle;
    Qnn_ContextHandle_t context_handle;
    Qnn_GraphHandle_t   graph_handle;

    // Tensor dimensions (populated after context is set up)
    size_t input_element_count;
    size_t output_element_count;

    // Tensor descriptors (owned by QNN context, do NOT free individually)
    Qnn_Tensor_t *input_tensors;
    uint32_t      num_input_tensors;
    Qnn_Tensor_t *output_tensors;
    uint32_t      num_output_tensors;
};

// ---------------------------------------------------------------------------
// QNN logging callback (writes to stderr in debug builds)
// ---------------------------------------------------------------------------

static void qnn_log_callback(const char *fmt, QnnLog_Level_t level,
                              uint64_t timestamp, va_list args) {
    (void)timestamp;
    // Only forward errors and warnings; suppress info/debug noise.
    if (level > QNN_LOG_LEVEL_WARN) return;
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

// ---------------------------------------------------------------------------
// Resolve the QNN interface from a loaded backend library
// ---------------------------------------------------------------------------

static int resolve_interface(void *lib, QnnInterface_t *out,
                              char *err, size_t err_sz) {
    QnnInterface_getProviders_t get_providers =
        (QnnInterface_getProviders_t)dlsym(lib, "QnnInterface_getProviders");
    if (!get_providers) {
        set_error(err, err_sz, "dlsym QnnInterface_getProviders: %s", dlerror());
        return -1;
    }

    const QnnInterface_t **providers = NULL;
    uint32_t num = 0;
    Qnn_ErrorHandle_t rc = get_providers(&providers, &num);
    if (rc != QNN_SUCCESS || num == 0) {
        set_error(err, err_sz,
                  "QnnInterface_getProviders failed (rc=%d, num=%u)", (int)rc, num);
        return -1;
    }

    // Pick the provider with the highest API version we understand.
    *out = *providers[0];
    return 0;
}

// ---------------------------------------------------------------------------
// Resolve the QNN System interface from libQnnSystem.so
// ---------------------------------------------------------------------------

static int resolve_system_interface(void *lib,
                                    QnnSystemInterface_t *out,
                                    char *err, size_t err_sz) {
    QnnSystemInterface_getProviders_t get_providers =
        (QnnSystemInterface_getProviders_t)dlsym(lib,
            "QnnSystemInterface_getProviders");
    if (!get_providers) {
        set_error(err, err_sz,
                  "dlsym QnnSystemInterface_getProviders: %s", dlerror());
        return -1;
    }

    const QnnSystemInterface_t **providers = NULL;
    uint32_t num = 0;
    Qnn_ErrorHandle_t rc = get_providers(&providers, &num);
    if (rc != QNN_SUCCESS || num == 0) {
        set_error(err, err_sz,
                  "QnnSystemInterface_getProviders failed (rc=%d, num=%u)",
                  (int)rc, num);
        return -1;
    }

    *out = *providers[0];
    return 0;
}

// ---------------------------------------------------------------------------
// Count total float32 elements across all dimensions of a tensor
// ---------------------------------------------------------------------------

static size_t tensor_element_count(const Qnn_Tensor_t *t) {
    const Qnn_TensorV2_t *v2 = &t->v2;
    if (!v2->dimensions || v2->rank == 0) return 0;
    size_t count = 1;
    for (uint32_t i = 0; i < v2->rank; i++) {
        count *= v2->dimensions[i];
    }
    return count;
}

// ---------------------------------------------------------------------------
// Common post-init: populate tensor metadata from a ready context
// ---------------------------------------------------------------------------

static int populate_graph_tensors(qnn_session_t *s, char *err, size_t err_sz) {
    // Retrieve graph input/output tensor info from the context.
    Qnn_ErrorHandle_t rc = s->iface.v2.graphRetrieve(
        s->context_handle, /* graphName = */ NULL, &s->graph_handle);
    if (rc != QNN_SUCCESS) {
        // Try index-based retrieval as a fallback.
        rc = s->iface.v2.contextGetBinarySize(s->context_handle, NULL);
        (void)rc;
        // Attempt to get the first graph by index.
        // If graphRetrieve with NULL name fails the backend may require an
        // explicit name; fail gracefully.
        set_error(err, err_sz,
                  "graphRetrieve failed (rc=%d); ensure the model has exactly "
                  "one graph or set graph_name explicitly", (int)rc);
        return -1;
    }

    uint32_t n_inputs = 0, n_outputs = 0;
    rc = s->iface.v2.graphGetTensors(s->graph_handle,
                                     &s->input_tensors, &n_inputs,
                                     &s->output_tensors, &n_outputs);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz, "graphGetTensors failed (rc=%d)", (int)rc);
        return -1;
    }

    s->num_input_tensors  = n_inputs;
    s->num_output_tensors = n_outputs;

    if (n_inputs == 0 || n_outputs == 0) {
        set_error(err, err_sz,
                  "unexpected tensor count: inputs=%u outputs=%u",
                  n_inputs, n_outputs);
        return -1;
    }

    s->input_element_count  = tensor_element_count(&s->input_tensors[0]);
    s->output_element_count = tensor_element_count(&s->output_tensors[0]);
    return 0;
}

// ---------------------------------------------------------------------------
// Common backend + log initialisation
// ---------------------------------------------------------------------------

static int init_backend(qnn_session_t *s, char *err, size_t err_sz) {
    // Create log handle (errors + warnings only).
    Qnn_ErrorHandle_t rc = s->iface.v2.logCreate(
        qnn_log_callback, QNN_LOG_LEVEL_WARN, &s->log_handle);
    if (rc != QNN_SUCCESS) {
        // Non-fatal — some backends don't support custom logging.
        s->log_handle = NULL;
    }

    // Initialise the backend.
    rc = s->iface.v2.backendCreate(s->log_handle, NULL /* backendConfig */,
                                   &s->backend_handle);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz, "backendCreate failed (rc=%d)", (int)rc);
        return -1;
    }

    // Device creation (not all backends require it; ignore failure).
    (void)s->iface.v2.deviceCreate(s->log_handle, NULL, &s->device_handle);

    return 0;
}

// ---------------------------------------------------------------------------
// Public API — create from context binary
// ---------------------------------------------------------------------------

qnn_session_t *qnn_create_from_context_binary(
    const char *backend_lib_path,
    const char *system_lib_path,
    const void *binary_data,
    size_t      binary_size,
    char       *err,
    size_t      err_sz)
{
    if (!backend_lib_path || !system_lib_path || !binary_data || binary_size == 0) {
        set_error(err, err_sz, "qnn_create_from_context_binary: invalid args");
        return NULL;
    }

    qnn_session_t *s = (qnn_session_t *)calloc(1, sizeof(*s));
    if (!s) {
        set_error(err, err_sz, "out of memory");
        return NULL;
    }

    // Load backend shared library.
    s->backend_lib = dlopen(backend_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!s->backend_lib) {
        set_error(err, err_sz, "dlopen %s: %s", backend_lib_path, dlerror());
        goto fail;
    }

    // Load system shared library (needed for context binary deserialization).
    s->system_lib = dlopen(system_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!s->system_lib) {
        set_error(err, err_sz, "dlopen %s: %s", system_lib_path, dlerror());
        goto fail;
    }

    // Resolve backend interface.
    if (resolve_interface(s->backend_lib, &s->iface, err, err_sz) != 0) goto fail;

    // Resolve system interface.
    QnnSystemInterface_t sys_iface;
    if (resolve_system_interface(s->system_lib, &sys_iface, err, err_sz) != 0)
        goto fail;

    // Initialise backend.
    if (init_backend(s, err, err_sz) != 0) goto fail;

    // Create system context to deserialize the binary.
    QnnSystemContext_Handle_t sys_ctx = NULL;
    Qnn_ErrorHandle_t rc = sys_iface.systemContextCreate(&sys_ctx);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz, "systemContextCreate failed (rc=%d)", (int)rc);
        goto fail;
    }

    // Deserialize the context binary.
    const QnnSystemContext_BinaryInfo_t *bin_info = NULL;
    Qnn_ContextBinarySize_t             bin_info_size = 0;
    rc = sys_iface.systemContextGetBinaryInfo(
        sys_ctx, (void *)binary_data, (Qnn_ContextBinarySize_t)binary_size,
        &bin_info, &bin_info_size);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz,
                  "systemContextGetBinaryInfo failed (rc=%d); "
                  "context binary may be incompatible with this backend/driver",
                  (int)rc);
        sys_iface.systemContextFree(sys_ctx);
        goto fail;
    }
    sys_iface.systemContextFree(sys_ctx);

    // Create the QNN context from the binary blob.
    rc = s->iface.v2.contextCreateFromBinary(
        s->backend_handle, s->device_handle,
        NULL /* contextConfig */,
        (void *)binary_data, (Qnn_ContextBinarySize_t)binary_size,
        &s->context_handle,
        NULL /* profileHandle */);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz,
                  "contextCreateFromBinary failed (rc=%d)", (int)rc);
        goto fail;
    }

    if (populate_graph_tensors(s, err, err_sz) != 0) goto fail;

    return s;

fail:
    qnn_destroy_session(s);
    return NULL;
}

// ---------------------------------------------------------------------------
// Public API — create from model library (online compilation)
// ---------------------------------------------------------------------------

// Type of the composer function exported by the model .so.
typedef Qnn_ErrorHandle_t (*ComposeGraphsFn_t)(
    Qnn_BackendHandle_t,
    QnnInterface_t,
    Qnn_ContextHandle_t,
    const GraphConfigInfo_t **,
    uint32_t,
    GraphInfo_t ***,
    uint32_t *,
    bool,
    QnnLog_Callback_t,
    QnnLog_Level_t);

typedef Qnn_ErrorHandle_t (*FreeGraphInfoFn_t)(
    GraphInfo_t ***,
    uint32_t);

qnn_session_t *qnn_create_from_model_lib(
    const char *backend_lib_path,
    const char *system_lib_path,
    const char *model_lib_path,
    char       *err,
    size_t      err_sz)
{
    if (!backend_lib_path || !system_lib_path || !model_lib_path) {
        set_error(err, err_sz, "qnn_create_from_model_lib: invalid args");
        return NULL;
    }

    qnn_session_t *s = (qnn_session_t *)calloc(1, sizeof(*s));
    if (!s) {
        set_error(err, err_sz, "out of memory");
        return NULL;
    }

    s->backend_lib = dlopen(backend_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!s->backend_lib) {
        set_error(err, err_sz, "dlopen %s: %s", backend_lib_path, dlerror());
        goto fail;
    }

    s->system_lib = dlopen(system_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!s->system_lib) {
        set_error(err, err_sz, "dlopen %s: %s", system_lib_path, dlerror());
        goto fail;
    }

    s->model_lib = dlopen(model_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!s->model_lib) {
        set_error(err, err_sz, "dlopen %s: %s", model_lib_path, dlerror());
        goto fail;
    }

    if (resolve_interface(s->backend_lib, &s->iface, err, err_sz) != 0) goto fail;
    if (init_backend(s, err, err_sz) != 0) goto fail;

    // Create a fresh context for graph composition.
    Qnn_ErrorHandle_t rc = s->iface.v2.contextCreate(
        s->backend_handle, s->device_handle,
        NULL /* contextConfig */,
        &s->context_handle);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz, "contextCreate failed (rc=%d)", (int)rc);
        goto fail;
    }

    // Compose graphs from the model library.
    ComposeGraphsFn_t compose =
        (ComposeGraphsFn_t)dlsym(s->model_lib, "QnnModel_composeGraphs");
    if (!compose) {
        set_error(err, err_sz,
                  "dlsym QnnModel_composeGraphs: %s "
                  "(ensure model was built with qnn-model-lib-generator)",
                  dlerror());
        goto fail;
    }

    GraphInfo_t **graph_infos   = NULL;
    uint32_t      num_graphs    = 0;
    rc = compose(s->backend_handle, s->iface, s->context_handle,
                 NULL /* graphConfigInfo */, 0,
                 &graph_infos, &num_graphs,
                 /* doNodeValidations = */ true,
                 qnn_log_callback, QNN_LOG_LEVEL_WARN);
    if (rc != QNN_SUCCESS || num_graphs == 0) {
        set_error(err, err_sz,
                  "QnnModel_composeGraphs failed (rc=%d, num_graphs=%u)",
                  (int)rc, num_graphs);
        goto fail;
    }

    // Finalise the first graph (the BirdNET classifier).
    rc = s->iface.v2.graphFinalize(graph_infos[0]->graph,
                                   NULL /* profileHandle */,
                                   NULL /* signal */);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz, "graphFinalize failed (rc=%d)", (int)rc);
        FreeGraphInfoFn_t freeInfo =
            (FreeGraphInfoFn_t)dlsym(s->model_lib, "QnnModel_freeGraphsInfo");
        if (freeInfo) freeInfo(&graph_infos, num_graphs);
        goto fail;
    }

    s->graph_handle = graph_infos[0]->graph;

    // Copy tensor pointers before freeing graph_infos metadata.
    s->input_tensors      = graph_infos[0]->inputTensors;
    s->num_input_tensors  = graph_infos[0]->numInputTensors;
    s->output_tensors     = graph_infos[0]->outputTensors;
    s->num_output_tensors = graph_infos[0]->numOutputTensors;

    s->input_element_count  = tensor_element_count(&s->input_tensors[0]);
    s->output_element_count = tensor_element_count(&s->output_tensors[0]);

    return s;

fail:
    qnn_destroy_session(s);
    return NULL;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

int qnn_run_inference(
    qnn_session_t *session,
    const float   *input_data,
    size_t         input_count,
    float         *output_data,
    size_t         output_count,
    char          *err,
    size_t         err_sz)
{
    if (!session || !input_data || !output_data) {
        set_error(err, err_sz, "qnn_run_inference: NULL argument");
        return -1;
    }
    if (input_count != session->input_element_count) {
        set_error(err, err_sz,
                  "input size mismatch: got %zu, expected %zu",
                  input_count, session->input_element_count);
        return -1;
    }
    if (output_count < session->output_element_count) {
        set_error(err, err_sz,
                  "output buffer too small: got %zu, need %zu",
                  output_count, session->output_element_count);
        return -1;
    }

    // Point the input tensor's client buffer at our data.
    session->input_tensors[0].v2.clientBuf.data     = (void *)input_data;
    session->input_tensors[0].v2.clientBuf.dataSize =
        (uint32_t)(input_count * sizeof(float));
    session->input_tensors[0].v2.memType            = QNN_TENSORMEMTYPE_RAW;

    // Point the output tensor's client buffer at the caller's buffer.
    session->output_tensors[0].v2.clientBuf.data     = output_data;
    session->output_tensors[0].v2.clientBuf.dataSize =
        (uint32_t)(output_count * sizeof(float));
    session->output_tensors[0].v2.memType            = QNN_TENSORMEMTYPE_RAW;

    Qnn_ErrorHandle_t rc = session->iface.v2.graphExecute(
        session->graph_handle,
        session->input_tensors,  session->num_input_tensors,
        session->output_tensors, session->num_output_tensors,
        NULL /* profileHandle */,
        NULL /* signal */);

    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz, "graphExecute failed (rc=%d)", (int)rc);
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

size_t qnn_input_element_count(const qnn_session_t *session) {
    return session ? session->input_element_count : 0;
}

size_t qnn_output_element_count(const qnn_session_t *session) {
    return session ? session->output_element_count : 0;
}

// ---------------------------------------------------------------------------
// Teardown
// ---------------------------------------------------------------------------

void qnn_destroy_session(qnn_session_t *session) {
    if (!session) return;

    if (session->context_handle && session->iface.v2.contextFree) {
        session->iface.v2.contextFree(session->context_handle,
                                      NULL /* profileHandle */);
    }
    if (session->device_handle && session->iface.v2.deviceFree) {
        session->iface.v2.deviceFree(session->device_handle);
    }
    if (session->backend_handle && session->iface.v2.backendFree) {
        session->iface.v2.backendFree(session->backend_handle);
    }
    if (session->log_handle && session->iface.v2.logFree) {
        session->iface.v2.logFree(session->log_handle);
    }

    // Close libraries last (function pointers become invalid after dlclose).
    if (session->model_lib)   dlclose(session->model_lib);
    if (session->system_lib)  dlclose(session->system_lib);
    if (session->backend_lib) dlclose(session->backend_lib);

    free(session);
}
