//go:build ncnn && linux && arm64 && !cross

package ncnn

/*
#cgo CFLAGS: -I/home/arduino/ncnn/build/install/include
#cgo LDFLAGS: -L/home/arduino/ncnn/build/install/lib -lncnn -lglslang -lMachineIndependent -lOSDependent -lGenericCodeGen -lSPIRV -lglslang-default-resource-limits -lvulkan -lstdc++ -lgomp -lpthread -lm -ldl
*/
import "C"
