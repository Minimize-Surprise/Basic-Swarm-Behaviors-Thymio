# imports for connection
import socket
from threading import Thread, Lock

class Client:

    def __init__(self):
        self.buffer = []
        self.mutex = Lock()
        self.stop_threads = False
        self.clientID = -1
        self.host_name = "MS0.local"  # TODO add master host name
        self.host_ip = socket.gethostbyname(self.host_name)
        self.port = 5555
        print("MASTER host_name: ", self.host_name)
        print("MASTER host_ip: ", self.host_ip)
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.clientSocket.connect((self.host_ip, self.port))  # verbindung zum server
        print("connected")

        message = "client connected..."
        self.sendMessage(message)

        try:
            self.com = Thread(target=self.recieveMessage)
            self.com.start()
        except:
            print("thread didnt start")

    # to send a message -> sends a string
    def sendMessage(self, msg):
        if self.clientID == -1:
            self.clientID = 0

        m = str(msg)
        self.clientSocket.send(m.encode())

    # to receive a message -> is in a dedicated thread
    def recieveMessage(self):
        global clientSocket
        global clientID
        global stop_threads

        while not self.stop_threads:
            try:
                msg = self.clientSocket.recv(1024).decode()
            except:
                print("something went wrong")
            if msg:
                if "clientindex" in msg:
                    # get clientID -> send by master
                    x = msg.replace("clientindex ", "")
                    print("CLIENTINDEX IS ", x)
                    self.clientID = int(x)
                else:
                    self.mutex.acquire()
                    self.buffer.append(str(msg))
                    self.mutex.release()

    def closeClient(self):
        self.stop_threads = True
        self.com.join()
        self.clientSocket.close()
        print("client closed...")


    def readBuffer(self):
        self.mutex.acquire()
        tmpBuffer = self.buffer.copy()
        self.buffer.clear()
        self.mutex.release()
        return tmpBuffer
