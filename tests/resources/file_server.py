from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000


class FileServer(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'audio/mpeg')
        self.send_header('Content-Disposition', 'filename="SampleAudio1.mp3"')
        self.end_headers()

        with open('test_audios/SampleAudio1.mp3', 'rb') as file:
            self.wfile.write(file.read())


HTTPServer(('0.0.0.0', PORT), FileServer).serve_forever()
