// fake_soc.c - LD_PRELOAD to fake /sys/devices/soc0/soc_id for QNN CPU backend
// Compile: aarch64-linux-gnu-gcc -shared -fPIC -o fake_soc.so fake_soc.c -ldl
// Usage: FAKE_SOC_ID=339 LD_PRELOAD=/tmp/fake_soc.so ./test_qnn9
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <errno.h>

// We intercept open/openat to detect reads to /sys/devices/soc0/soc_id
// When that file is opened, we return a pipe FD that returns our fake value

static const char* SOC_PATH = "/sys/devices/soc0/soc_id";
static int fake_pipe_read = -1;
static int fake_pipe_write = -1;
static char fake_soc_str[32] = "356\n";  // Default: SM8250 (Snapdragon 865)

static void init_fake(void) {
    const char* env = getenv("FAKE_SOC_ID");
    if (env) {
        snprintf(fake_soc_str, sizeof(fake_soc_str), "%s\n", env);
    }
}

static __attribute__((constructor)) void ctor(void) {
    init_fake();
    fprintf(stderr, "[fake_soc] Loaded, will fake soc_id as: %s", fake_soc_str);
}

// Override open()
int open(const char* pathname, int flags, ...) {
    static int (*real_open)(const char*, int, ...) = NULL;
    if (!real_open) real_open = dlsym(RTLD_NEXT, "open");

    if (pathname && strcmp(pathname, SOC_PATH) == 0) {
        int pfd[2];
        if (pipe(pfd) == 0) {
            // Write our fake soc_id to the write end
            write(pfd[1], fake_soc_str, strlen(fake_soc_str));
            close(pfd[1]);
            fprintf(stderr, "[fake_soc] Intercepted open(%s) -> returning fake fd with '%s'",
                SOC_PATH, fake_soc_str);
            return pfd[0];  // Return read end
        }
    }

    va_list args;
    va_start(args, flags);
    mode_t mode = va_arg(args, mode_t);
    va_end(args);
    return real_open(pathname, flags, mode);
}

// Override openat()
int openat(int dirfd, const char* pathname, int flags, ...) {
    static int (*real_openat)(int, const char*, int, ...) = NULL;
    if (!real_openat) real_openat = dlsym(RTLD_NEXT, "openat");

    if (pathname && strcmp(pathname, SOC_PATH) == 0) {
        int pfd[2];
        if (pipe(pfd) == 0) {
            write(pfd[1], fake_soc_str, strlen(fake_soc_str));
            close(pfd[1]);
            fprintf(stderr, "[fake_soc] Intercepted openat(%s) -> returning fake fd with '%s'",
                SOC_PATH, fake_soc_str);
            return pfd[0];
        }
    }

    va_list args;
    va_start(args, flags);
    mode_t mode = va_arg(args, mode_t);
    va_end(args);
    return real_openat(dirfd, pathname, flags, mode);
}
