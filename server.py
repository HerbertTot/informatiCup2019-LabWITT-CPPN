# -*- coding: utf-8 -*-
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.endpoints import WebSocketEndpoint

import uvicorn
import re
import aiohttp
import base64
from skimage import io
from starlette.staticfiles import StaticFiles

from threading import Thread

from util import send_query, load_image, load_img_to_bytes, image_to_grayscale
from generate_adversarial import generate_adversarial


app = Starlette(debug=True, template_directory='server/templates')
app.mount('/static', StaticFiles(directory='server/static'), name='static')


@app.route("/")
async def progress(request):
    template = app.get_template('index.html')
    content = template.render(request=request)

    return HTMLResponse(content)


@app.websocket_route('/')
class Socket(WebSocketEndpoint):
    encoding = 'json'

    def __init__(self, scope):
        self.id = None
        super().__init__(scope)

    async def on_connect(self, websocket):
        await websocket.accept()

    async def on_receive(self, websocket, data):
        # If the id tag is send with other data, we assume the client wants to set his ID
        if 'id' in data.keys():
            print('Client with ID', data['id'], 'connected')

            self.id = data['id']

        if 'image' in data.keys():
            print('Recived an Image')

            # Strip die data\png;base64, from the image data stream
            matcher = re.match(r"data:([\w\/\+]+);(charset=[\w-]+|base64).*,([a-zA-Z0-9+/]+={0,2})", data['image'])
            raw_data = matcher.group(3)

            # Save file with the id of the client
            with open('server/image/' + self.id, "wb") as f:
                f.write(base64.b64decode(raw_data))

            # Reload Image as Byte object. It is also croped to size
            img = load_img_to_bytes('server/image/' + self.id)
            img = io.imread(img)

            # Get Labels for the Image from the API
            labels, confs = send_query(img)

            # Send Labels to the the Client
            await websocket.send_json({'labels': labels.tolist(), 'confs': confs.tolist()})

        if {'start', 'target_class', 'target_conf', 'init', 'max_queries', 'rgb'} <= data.keys():
            if data['start']:
                print('Start fooling with',
                      data['target_class'],
                      data['target_conf'],
                      data['max_queries'],
                      data['init'],
                      data['rgb'],
                      'for client',
                      self.id)

                img = load_image('server/image/' + self.id, size=64)

                if not data['rgb']:
                    img = image_to_grayscale(img)
                thread = Thread(target=generate_adversarial, kwargs=dict(target_class=data['target_class'],
                                                                         target_image=img,
                                                                         target_conf=float(data['target_conf']),
                                                                         init=data['init'],
                                                                         max_queries=int(data['max_queries']),
                                                                         color=data['rgb'],
                                                                         websocket=websocket,
                                                                         client_id=self.id))
                thread.start()

    async def on_disconnect(self, websocket, close_code):
        print('Connection closed to client with id:', self.id, 'and Code:', close_code)
        await websocket.close()


uvicorn.run(app, host="127.0.0.1", port=8008)
