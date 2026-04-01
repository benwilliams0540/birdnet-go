// qnn_backend.h — thin C shim over the Qualcomm QNN C API.
//
// This layer loads the QNN backend library (e.g. libQnnGpu.so or libQnnHtp.so)
// and the QNN system library (libQnnSystem.so) at runtime via dlopen, so the
// birdnet-go binary itself has no link-time dependency on the QAIRT SDK.
//
// The caller is responsible for:
//   1. Running qnn-onnx-converter + qnn-model-lib-generator offline to produce
//      a compiled model shared library (libmodel_net.so) from the ONNX file.
//   2. Optionally running qnn-context-binary-generator on the target device to
//      produce a backend-specific context binary (.bin) for faster start-up.
//   3. Deploying libQnnGpu.so (or libQnnHtp.so), libQnnSystem.so, and the model
//      artifacts alongside the birdnet-go binary.
//
// Two loading modes are supported:
//   a) Context binary  — fastest start, device/driver specific.
//   b) Model library   — portable, JIT-compiled by the backend on first use.
//      (HTP only) Requires libQnnHtpPrepare.so and the HTP stub .so files.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle returned by qnn_create_*.
typedef struct qnn_session qnn_session_t;

// --------------------------------------------------------------------------
// Session creation
// --------------------------------------------------------------------------

// qnn_create_from_context_binary loads a pre-compiled QNN context binary.
//
//   backend_lib_path  – absolute path to libQnnGpu.so or libQnnHtp.so
//   system_lib_path   – absolute path to libQnnSystem.so
//   binary_data       – raw bytes of the context binary (.bin file)
//   binary_size       – byte count
//   error_buf         – caller-allocated buffer for error messages (may be NULL)
//   error_buf_size    – size of error_buf in bytes
//
// Returns a non-NULL session on success, NULL on failure (see error_buf).
qnn_session_t *qnn_create_from_context_binary(
    const char  *backend_lib_path,
    const char  *system_lib_path,
    const void  *binary_data,
    size_t       binary_size,
    char        *error_buf,
    size_t       error_buf_size
);

// qnn_create_from_model_lib loads a QNN model library and performs online
// graph composition + backend compilation.  Slower first run but portable
// across GPU driver versions.
//
//   backend_lib_path  – absolute path to libQnnGpu.so or libQnnHtp.so
//   system_lib_path   – absolute path to libQnnSystem.so
//   model_lib_path    – absolute path to the compiled model .so
//                       (output of qnn-model-lib-generator, e.g. libmodel_net.so)
//   error_buf / error_buf_size – as above
qnn_session_t *qnn_create_from_model_lib(
    const char  *backend_lib_path,
    const char  *system_lib_path,
    const char  *model_lib_path,
    char        *error_buf,
    size_t       error_buf_size
);

// --------------------------------------------------------------------------
// Inference
// --------------------------------------------------------------------------

// qnn_run_inference executes one forward pass synchronously.
//
//   session      – handle from qnn_create_*
//   input_data   – flat float32 input buffer (must match model input size)
//   input_count  – number of float elements in input_data
//   output_data  – caller-allocated output buffer
//   output_count – number of float elements expected in output_data
//
// Returns 0 on success, non-zero on failure (see error_buf).
int qnn_run_inference(
    qnn_session_t *session,
    const float   *input_data,
    size_t         input_count,
    float         *output_data,
    size_t         output_count,
    char          *error_buf,
    size_t         error_buf_size
);

// --------------------------------------------------------------------------
// Introspection
// --------------------------------------------------------------------------

// Returns the number of float32 elements expected in the model input tensor.
size_t qnn_input_element_count(const qnn_session_t *session);

// Returns the number of float32 elements produced in the model output tensor.
size_t qnn_output_element_count(const qnn_session_t *session);

// --------------------------------------------------------------------------
// Clean-up
// --------------------------------------------------------------------------

// qnn_destroy_session releases all QNN resources and closes the loaded
// shared libraries.  Passing NULL is a no-op.
void qnn_destroy_session(qnn_session_t *session);

#ifdef __cplusplus
}
#endif
