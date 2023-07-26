package main

import (
	"io"
	"time"
)

// PingWriter simulates a delay before each write of a writer
type PingWriter struct {
	w    io.Writer
	ping time.Duration
}

func (w PingWriter) Write(data []byte) (int, error) {
	time.Sleep(w.ping)
	return w.w.Write(data)
}
