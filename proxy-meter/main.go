package main

import (
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
)

type ProxiedConnections struct {
	c1, c2 net.Conn
}

// A counter to give ID to each connection
var connectionIDCounter atomic.Uint32

// connectionMap is a map from the connection ID to both sides of connections
var connectionMap = make(map[uint32]ProxiedConnections)
var connectionMapMutex sync.Mutex

// How many active connections we have?
var activeConnectionCounter sync.WaitGroup

func main() {
	// Read values
	listenAddress := os.Getenv("PROXY_LISTEN")
	if listenAddress == "" {
		log.Fatalln("Please set PROXY_LISTEN environment variable")
	}
	forwardAddress := os.Getenv("PROXY_FORWARD")
	if forwardAddress == "" {
		log.Fatalln("Please set PROXY_FORWARD environment variable")
	}
	// Start listener
	l, err := net.Listen("tcp", listenAddress)
	if err != nil {
		log.Fatalln("cannot listen:", err)
	}
	// Hook closer
	signalChannel := make(chan os.Signal, 1)
	signal.Notify(signalChannel, os.Interrupt)
	go func() {
		<-signalChannel
		_ = l.Close()
	}()
	// Wait for connections
	for {
		conn, err := l.Accept()
		if err != nil {
			log.Println("cannot accept connection:", err)
			break
		}
		connID := connectionIDCounter.Add(1)
		// Connect to remote
		remoteConn, err := net.Dial("tcp", forwardAddress)
		if err != nil {
			log.Println("cannot dial remote:", err)
			break
		}
		// Add to list
		connectionMapMutex.Lock()
		connectionMap[connID] = ProxiedConnections{conn, remoteConn}
		activeConnectionCounter.Add(2) // add two, because we are creating two goroutines
		connectionMapMutex.Unlock()
		// Proxy both ways
		go proxyConnection(conn, remoteConn, "download", connID)
		go proxyConnection(remoteConn, conn, "upload", connID)
	}
	// Close active connections
	connectionMapMutex.Lock()
	for _, connections := range connectionMap {
		_ = connections.c1.Close()
		_ = connections.c2.Close()
	}
	connectionMapMutex.Unlock()
	// Wait for all of them
	activeConnectionCounter.Wait()
}

func proxyConnection(c1 net.Conn, c2 net.Conn, tag string, id uint32) {
	copied, _ := io.Copy(c1, c2)
	log.Printf("%s connection %d finished with %d bytes trasfered\n", tag, id, copied)
	connectionMapMutex.Lock()
	delete(connectionMap, id)
	_ = c1.Close()
	_ = c2.Close()
	connectionMapMutex.Unlock()
	activeConnectionCounter.Done()
}
