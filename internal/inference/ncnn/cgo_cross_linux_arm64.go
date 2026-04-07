//go:build ncnn && linux && arm64 && cross

package ncnn

/*
#cgo LDFLAGS: -lncnn -lglslang -lMachineIndependent -lOSDependent -lGenericCodeGen -lSPIRV -lglslang-default-resource-limits -lvulkan -lstdc++ -lgomp -lpthread -lm -ldl
*/
import "C"
