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
#include <stdbool.h>

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
static int extract_tensors_from_metadata(qnn_session_t *s, const QnnSystemContext_BinaryInfo_t *bin_info, char *err, size_t err_sz) {
    if (!bin_info) return -1;
    const QnnSystemContext_GraphInfo_t* graph_info = NULL;

    if (bin_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        if (bin_info->contextBinaryInfoV1.numGraphs == 0) return -1;
        graph_info = &bin_info->contextBinaryInfoV1.graphs[0];
    } else if (bin_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        if (bin_info->contextBinaryInfoV2.numGraphs == 0) return -1;
        graph_info = &bin_info->contextBinaryInfoV2.graphs[0];
    } else {
        if (bin_info->contextBinaryInfoV3.numGraphs == 0) return -1;
        graph_info = &bin_info->contextBinaryInfoV3.graphs[0];
    }

    uint32_t num_in = 0, num_out = 0;
    Qnn_Tensor_t *in_tensors = NULL, *out_tensors = NULL;
    const char* graph_name = NULL;

    if (graph_info->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        num_in = graph_info->graphInfoV1.numGraphInputs;
        in_tensors = graph_info->graphInfoV1.graphInputs;
        num_out = graph_info->graphInfoV1.numGraphOutputs;
        out_tensors = graph_info->graphInfoV1.graphOutputs;
        graph_name = graph_info->graphInfoV1.graphName;
    } else if (graph_info->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        num_in = graph_info->graphInfoV2.numGraphInputs;
        in_tensors = graph_info->graphInfoV2.graphInputs;
        num_out = graph_info->graphInfoV2.numGraphOutputs;
        out_tensors = graph_info->graphInfoV2.graphOutputs;
        graph_name = graph_info->graphInfoV2.graphName;
    } else {
        num_in = graph_info->graphInfoV3.numGraphInputs;
        in_tensors = graph_info->graphInfoV3.graphInputs;
        num_out = graph_info->graphInfoV3.numGraphOutputs;
        out_tensors = graph_info->graphInfoV3.graphOutputs;
        graph_name = graph_info->graphInfoV3.graphName;
    }

    if (num_in == 0 || num_out == 0) {
        set_error(err, err_sz, "unexpected tensor count");
        return -1;
    }

    s->num_input_tensors = num_in;
    s->num_output_tensors = num_out;
    s->input_tensors = calloc(num_in, sizeof(Qnn_Tensor_t));
    s->output_tensors = calloc(num_out, sizeof(Qnn_Tensor_t));
    for (uint32_t i=0; i<num_in; ++i) s->input_tensors[i] = in_tensors[i];
    for (uint32_t i=0; i<num_out; ++i) s->output_tensors[i] = out_tensors[i];

    s->input_element_count = 0;
    for (uint32_t i = 0; i < s->num_input_tensors; ++i)
        s->input_element_count += tensor_element_count(&s->input_tensors[i]);
    s->output_element_count = tensor_element_count(&s->output_tensors[0]);

    if (graph_name) {
        s->iface.v2_29.graphRetrieve(s->context_handle, graph_name, &s->graph_handle);
    } else {
        s->iface.v2_29.graphRetrieve(s->context_handle, NULL, &s->graph_handle);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Common backend + log initialisation
// ---------------------------------------------------------------------------

static int init_backend(qnn_session_t *s, char *err, size_t err_sz) {
    // Create log handle (errors + warnings only).
    Qnn_ErrorHandle_t rc = s->iface.v2_29.logCreate(
        qnn_log_callback, QNN_LOG_LEVEL_WARN, &s->log_handle);
    if (rc != QNN_SUCCESS) {
        // Non-fatal — some backends don't support custom logging.
        s->log_handle = NULL;
    }

    // Initialise the backend.
    rc = s->iface.v2_29.backendCreate(s->log_handle, NULL /* backendConfig */,
                                   &s->backend_handle);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz, "backendCreate failed (rc=%d)", (int)rc);
        return -1;
    }

    // Device creation (not all backends require it; ignore failure).
    (void)s->iface.v2_29.deviceCreate(s->log_handle, NULL, &s->device_handle);

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
    Qnn_ErrorHandle_t rc = sys_iface.v1_5.systemContextCreate(&sys_ctx);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz, "systemContextCreate failed (rc=%d)", (int)rc);
        goto fail;
    }

    // Deserialize the context binary.
    const QnnSystemContext_BinaryInfo_t *bin_info = NULL;
    Qnn_ContextBinarySize_t             bin_info_size = 0;
    rc = sys_iface.v1_5.systemContextGetBinaryInfo(
        sys_ctx, (void *)binary_data, (Qnn_ContextBinarySize_t)binary_size,
        &bin_info, &bin_info_size);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz,
                  "systemContextGetBinaryInfo failed (rc=%d); "
                  "context binary may be incompatible with this backend/driver",
                  (int)rc);
        sys_iface.v1_5.systemContextFree(sys_ctx);
        goto fail;
    }
    // Create the QNN context from the binary blob.
    rc = s->iface.v2_29.contextCreateFromBinary(
        s->backend_handle, s->device_handle,
        NULL /* contextConfig */,
        (void *)binary_data, (Qnn_ContextBinarySize_t)binary_size,
        &s->context_handle,
        NULL /* profileHandle */);
    if (rc != QNN_SUCCESS) {
        set_error(err, err_sz,
                  "contextCreateFromBinary failed (rc=%d)", (int)rc);
        sys_iface.v1_5.systemContextFree(sys_ctx);
        goto fail;
    }

    if (extract_tensors_from_metadata(s, bin_info, err, err_sz) != 0) {
        sys_iface.v1_5.systemContextFree(sys_ctx);
        goto fail;
    }

    sys_iface.v1_5.systemContextFree(sys_ctx);

    return s;

fail:
    qnn_destroy_session(s);
    return NULL;
}

// ---------------------------------------------------------------------------
// Public API — create from model library (online compilation)
// ---------------------------------------------------------------------------

typedef struct {
    char* graphName;
    const QnnGraph_Config_t** graphConfigs;
} GraphConfigInfo_t;

typedef struct {
    Qnn_GraphHandle_t graph;
    char* graphName;
    Qnn_Tensor_t* inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t* outputTensors;
    uint32_t numOutputTensors;
} GraphInfo_t;

// Type of the composer function exported by the model .so.
// Second arg is QNN_INTERFACE_VER_TYPE (the impl struct), NOT the full QnnInterface_t.
typedef int (*ComposeGraphsFn_t)(
    Qnn_BackendHandle_t,
    QNN_INTERFACE_VER_TYPE,
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

    // Load with RTLD_GLOBAL so the backend's symbols are visible to the model
    // library and any internal QNN subsystems that rely on global symbol lookup
    // (verified: RTLD_LOCAL causes SIGSEGV in composeGraphs on Debian/aarch64).
    s->backend_lib = dlopen(backend_lib_path, RTLD_NOW | RTLD_GLOBAL);
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
    Qnn_ErrorHandle_t rc = s->iface.v2_29.contextCreate(
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
    rc = compose(s->backend_handle, s->iface.QNN_INTERFACE_VER_NAME, s->context_handle,
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
    rc = s->iface.v2_29.graphFinalize(graph_infos[0]->graph,
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
    s->num_input_tensors  = graph_infos[0]->numInputTensors;
    s->num_output_tensors = graph_infos[0]->numOutputTensors;
    s->input_tensors = calloc(s->num_input_tensors, sizeof(Qnn_Tensor_t));
    s->output_tensors = calloc(s->num_output_tensors, sizeof(Qnn_Tensor_t));
    for (uint32_t i=0; i<s->num_input_tensors; ++i) s->input_tensors[i] = graph_infos[0]->inputTensors[i];
    for (uint32_t i=0; i<s->num_output_tensors; ++i) s->output_tensors[i] = graph_infos[0]->outputTensors[i];

    s->input_element_count = 0;
    for (uint32_t i = 0; i < s->num_input_tensors; ++i)
        s->input_element_count += tensor_element_count(&s->input_tensors[i]);
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

    // Point each input tensor's client buffer at the appropriate slice of input_data.
    // For single-input models input_count == tensor_element_count(&input_tensors[0]);
    // for the two-input CNN model the caller concatenates [SPEC1 | SPEC2] into one array.
    {
        size_t offset = 0;
        for (uint32_t i = 0; i < session->num_input_tensors; ++i) {
            size_t n = tensor_element_count(&session->input_tensors[i]);
            session->input_tensors[i].v2.clientBuf.data     = (void *)(input_data + offset);
            session->input_tensors[i].v2.clientBuf.dataSize = (uint32_t)(n * sizeof(float));
            session->input_tensors[i].v2.memType            = QNN_TENSORMEMTYPE_RAW;
            offset += n;
        }
    }

    // Point the output tensor's client buffer at the caller's buffer.
    session->output_tensors[0].v2.clientBuf.data     = output_data;
    session->output_tensors[0].v2.clientBuf.dataSize =
        (uint32_t)(output_count * sizeof(float));
    session->output_tensors[0].v2.memType            = QNN_TENSORMEMTYPE_RAW;

    Qnn_ErrorHandle_t rc = session->iface.v2_29.graphExecute(
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

    if (session->context_handle && session->iface.v2_29.contextFree) {
        session->iface.v2_29.contextFree(session->context_handle,
                                      NULL /* profileHandle */);
    }
    if (session->device_handle && session->iface.v2_29.deviceFree) {
        session->iface.v2_29.deviceFree(session->device_handle);
    }
    if (session->backend_handle && session->iface.v2_29.backendFree) {
        session->iface.v2_29.backendFree(session->backend_handle);
    }
    if (session->log_handle && session->iface.v2_29.logFree) {
        session->iface.v2_29.logFree(session->log_handle);
    }

    if (session->input_tensors) free(session->input_tensors);
    if (session->output_tensors) free(session->output_tensors);

    // Close libraries last (function pointers become invalid after dlclose).
    if (session->model_lib)   dlclose(session->model_lib);
    if (session->system_lib)  dlclose(session->system_lib);
    if (session->backend_lib) dlclose(session->backend_lib);

    free(session);
}
