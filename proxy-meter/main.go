package main

import (
	"github.com/juju/ratelimit"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
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

// How much should we wait before actually proxying data
var pingDelay time.Duration
var speedLimit int64

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
	pingDelay, _ = time.ParseDuration(os.Getenv("PROXY_DELAY"))
	speedLimit, _ = strconv.ParseInt(os.Getenv("PROXY_SPEED_LIMIT"), 10, 64)
	// Log
	log.Printf("Starting proxy\n")
	log.Printf("Forwarding from %s to %s\n", listenAddress, forwardAddress)
	log.Printf("Simulating a ping delay of %s\n", pingDelay)
	log.Printf("Proxy has a speedlimit of %d bytes/second\n", speedLimit)
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
		// Simulate ping
		time.Sleep(pingDelay)
		// Proxy both ways
		go proxyConnection(conn, remoteConn, "download", conn.RemoteAddr().String(), connID)
		go proxyConnection(remoteConn, conn, "upload", conn.RemoteAddr().String(), connID)
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

func proxyConnection(c1 net.Conn, c2 net.Conn, tag, localAddress string, id uint32) {
	// Speed limit if needed
	var copied int64
	if speedLimit == 0 {
		copied, _ = io.Copy(c1, c2)
	} else {
		copied, _ = io.Copy(ratelimit.Writer(c1, getBucket()), c2)
	}
	log.Printf("%s connection %s (%d) finished with %d bytes trasfered\n", tag, localAddress, id, copied)
	connectionMapMutex.Lock()
	delete(connectionMap, id)
	_ = c1.Close()
	_ = c2.Close()
	connectionMapMutex.Unlock()
	activeConnectionCounter.Done()
}

// getBucket will get a rate limit bucket which fills at the speed of
// speedLimit bytes per second
func getBucket() *ratelimit.Bucket {
	// Fill the bucket at speed of 1MB per second
	return ratelimit.NewBucketWithRate(float64(speedLimit), speedLimit)
}
