from http.server import BaseHTTPRequestHandler, HTTPServer
import json

def MakeHandlerClassFromArgv(init_args):
    class CustomHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
             self.agent=init_args
             super(CustomHandler, self).__init__(*args, **kwargs)

        def log_message(self, format, *args):
            return
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            agenReq=""
            with self.agent.lock:
                if len(self.agent.agentRequests):
                    agenReq=self.agent.agentRequests[0]
            self.wfile.write(bytes(json.dumps(agenReq), 'utf-8'))

        def do_POST(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = self.rfile.read(int(self.headers['Content-Length']))

            data = json.loads(data)
            if not len(self.agent.agentRequests):
                raise Exception('self.agent.agentRequests is empty')
            with self.agent.cv:
                self.agent.data=data
                # print(data)
                self.agent.cv.notify()
                self.agent.agentRequests.pop(0)

    return CustomHandler
