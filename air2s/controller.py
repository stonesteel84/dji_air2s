import socket
import cv2

class Air2SException(RuntimeError):
    pass

class Air2S:
    def __init__(self, ip='192.168.42.1', cmd_port=8889, video_port=11111):
        self.ip = ip
        self.cmd_port = cmd_port
        self.video_port = video_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(5)

    def _send(self, cmd: str) -> str:
        self.sock.sendto(cmd.encode(), (self.ip, self.cmd_port))
        resp, _ = self.sock.recvfrom(1024)
        return resp.decode().strip()

    def connect(self):
        self._send('command')

    def get_battery(self) -> int:
        return int(self._send('battery?'))

    def streamon(self):
        self._send('streamon')

    def streamoff(self):
        self._send('streamoff')

    def get_frame_read(self):
        cap = cv2.VideoCapture(f'udp://@0.0.0.0:{self.video_port}')
        class Reader:
            def __init__(self, cap):
                self.cap = cap
            @property
            def frame(self):
                ok, f = self.cap.read()
                return f if ok else None
        return Reader(cap)

    def takeoff(self):
        self._send('takeoff')

    def land(self):
        self._send('land')

    def emergency(self):
        self._send('emergency')

    def end(self):
        self.sock.close()

    def send_rc_control(self, lr, fb, ud, yaw):
        self._send(f'rc {lr} {fb} {ud} {yaw}')

    def get_height(self) -> int:
        return int(self._send('height?'))
