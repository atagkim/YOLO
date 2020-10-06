from pythreejs import *
import numpy as np
from IPython.display import display
from ipywidgets import HTML, Text, Output, VBox
from traitlets import link, dlink

ball = Mesh(geometry=SphereGeometry(radius=1, widthSegments=32, heightSegments=24), 
            material=MeshLambertMaterial(color='red'),
            position=[2, 1, 0])

c = PerspectiveCamera(position=[0, 5, 5], up=[0, 1, 0],
                      children=[DirectionalLight(color='white', position=[3, 5, 1], intensity=0.5)])

scene = Scene(children=[ball, c, AmbientLight(color='#777777')])

renderer = Renderer(camera=c, 
                    scene=scene, 
                    controls=[OrbitControls(controlling=c)])
display(renderer)