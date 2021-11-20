# imports for connection
import socket
from threading import Thread, Lock
# global mutex = Lock()

class Server:

    def __init__(self):
        self.clientList = []
        self.stop_threads = False
        self.buffer = []
        self.mutex = Lock()

        try:
            self.com = Thread(target=self.dispatcher)
            self.com.start()
        except:
            print("thread didnt start")

    def dispatcher(self):

        self.clientList = []
        self.masterSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host_name = socket.gethostname()  # gibt hostname
        self.masterSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.port = 5555  # Reserve a port fo r your service.
        clientIndex = 0

        print("MASTER host_name: ", self.host_name)
        print("MASTER Port: ", self.port)

        print('Server started!')
        print('Waiting for clients...')

        self.masterSocket.bind(('', self.port))  # Bind to the port

        # _thread.start_new_thread(dispatcher())
        self.masterSocket.listen(5)  # Now wait for client connection.
        while not self.stop_threads:
            conn, addr = self.masterSocket.accept()  # Establish connection with client.
            con = Thread(target=self.on_new_client, args=(conn, addr, clientIndex))
            con.start()
            self.clientList.append(conn)
            clientIndex = clientIndex + 1

    # receiving from clients
    def on_new_client(self, clientsocket, addr, clientIndex):
    
        print("connected to new client... ")
        s = "clientindex " + str(clientIndex)
        clientsocket.send(s.encode())

        # communication receiving from clients
        msg = ""
        while msg != "close" or not self.stop_threads:  # disconnect via close
            # receive msg from client
            try:
                msg = clientsocket.recv(1024).decode()
            except:
                self.clientList.remove(clientsocket)
                clientsocket.close()
                print("something with accepting went wrong")

            if not msg:
                break

            try:
                self.mutex.acquire()
                self.buffer.append(str(msg))
                self.mutex.release()
            except ValueError:
                print("wrong type")

    # sending broadcast to all registrated clients
    def sendBroadcast(self, var):
        for client in self.clientList:
            client.send(var.encode())

    def closeServer(self):
        self.stop_threads = True
        tmpSocket = socket.socket()
        tmpSocket.connect(("127.0.0.1", 5555))
        self.com.join()
        tmpSocket.close()

    def readBuffer(self):
        self.mutex.acquire()
        tmpBuffer = self.buffer.copy()
        self.buffer.clear()
        self.mutex.release()
        return tmpBuffer
