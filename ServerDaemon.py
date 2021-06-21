from http.server import BaseHTTPRequestHandler, HTTPServer
from gym_roblox.envs.RobloxPendulum import RobloxPendulum as Agent
import json,time,queue,threading,logging
import webbrowser


EnvironmentResponse=queue.Queue()
hostName = 'localhost'
serverPort = 8080

agent=Agent()

class MyServer(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        agenReq=""
        with agent.lock:
            if len(agent.agentRequests):
                agenReq=agent.agentRequests[0]
        self.wfile.write(bytes(json.dumps(agenReq), 'utf-8'))

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        # time.sleep(1)
        data = self.rfile.read(int(self.headers['Content-Length']))

        data = json.loads(data)
        if not len(agent.agentRequests):
            raise Exception('agent.agentRequests is empty')
        with agent.cv:
            agent.data=data
            agent.cv.notify()
            print(data)
            agent.agentRequests.pop(0)


def test_agent():
    for i in range(20):
        agent.reset()
        while(not agent.done):
            agent.step([0])
        print("==========================================")


if __name__ == '__main__':
    server = HTTPServer((hostName, serverPort), MyServer)
    print(f'Server started http://{hostName}:{serverPort}')
    otherThread=threading.Thread(name='tester', target=test_agent,args=())
    thread = threading.Thread(target = server.serve_forever)
    thread.daemon = True
    thread.start()
    test_agent()

    print('Server stopped.')
