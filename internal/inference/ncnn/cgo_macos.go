//go:build ncnn && darwin

package ncnn

/*
#cgo pkg-config: ncnn
#cgo LDFLAGS: -lstdc++ -lpthread -lm -ldl
*/
import "C"
