import socket
import time
from tinyrpc.transports import ClientTransport

class TcpTransport(ClientTransport):
    def __init__(self, port, host = '127.0.0.1') -> None:
        self.prev_buffer = ''

        retry = 0
        while True:
            try:
                # If you see crash here, hit Play in RobloxStudio first
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((host, port))
                break
            except socket.error as e:
                retry += 1
                if retry > 10000:
                    raise
                time.sleep(0.1)

    """Abstract base class for all client transports.

    The client side implementation of the transport component.
    Requests and replies encoded by the protocol component are
    exchanged between client and server using the :py:class:`ServerTransport`
    and :py:class:`ClientTransport` classes.    """
    def send_message(self, message: bytes, expect_reply: bool = True) -> bytes:
        """Send a message to the server and possibly receive a reply.

        Sends a message to the connected server.

        The message must be treated as a binary entity as only the protocol
        level will know how to interpret the message.

        If the transport encodes the message in some way, the opposite end
        is responsible for decoding it before it is passed to either client
        or server.

        This function will block until the reply has been received.

        :param bytes message: The request to send to the server.
        :param bool expect_reply: Some protocols allow notifications for which a
            reply is not expected. When this flag is ``False`` the transport may
            not wait for a response from the server.
            **Note** that it is still the responsibility of the transport layer how
            to implement this. It is still possible that the server sends some form
            of reply regardless the value of this flag.
        :return: The servers reply to the request.
        :rtype: bytes
        """
        message += b'\n'
        self.socket.sendall(message)

        if not expect_reply:
            return

        # The socket have data ready to be received
        buffer = self.prev_buffer
        self.prev_buffer = ''
        continue_recv = True

        result = ''
        while continue_recv:
            try:
                # Try to receive som data
                buffer += self.socket.recv(1024).decode()
                pos = buffer.find('\n')
                if pos != -1:
                    result = buffer[0:pos]
                    self.prev_buffer = buffer[pos + 1:]

                    continue_recv = False
            except socket.error as e:
                if e.errno != socket.errno.EWOULDBLOCK:
                    # Error! Print it and tell main loop to stop
                    print('Error: %r' % e)
                    return ''
                # If e.errno is errno.EWOULDBLOCK, then no more data
                continue_recv = False
        return result