from getlattice import generate_cube_points
import numpy as np

def emitJson2Part():
    squareCenter=emitjson[0]["squareCenter"]
    squareCenter=np.array(squareCenter)
    squareCenter*=scale

    points_emit=generate_cube_points(
        center=squareCenter,
        length=emitjson[0]["squareSize"][0]*scale,
        height=emitjson[0]["squareSize"][2]*scale,
        width= emitjson[0]["squareSize"][1]*scale,
        spacing=0.05
    )
    vel_emit=np.ones_like(points_emit) * emitjson[0]["velocity"]

    return points_emit,vel_emit


def get_partemit_wavetower(scale=1):
    range_=range(0, 1, 1)

    emitjson=[
        {
        "objectId": 0,
     	"squareCenter":[5.05, 0.525, 1.5],
		"squareSize": [1.7, 0.9, 2.8],
        "translation": [0.0, 0.0, 0.0],
        "scale": [1, 1, 1],
        "velocity": [0.0, 0.0, 0.0],
        "density": 1000.0,
        "color": [50, 100, 200]
    }
    ]


 
 
    return emitJson2Part(emitjson),range_



def get_partemit_watervessel(scale=1):

    range_=range(0, 1, 1)

    emitjson=[
        {
        "objectId": 0,
        "squareCenter": [3, 0.25, 3],
        "squareSize": [5.8, 0.3, 5.8],
        "translation": [0.0, 0.0, 0.0],
        "scale": [1, 1, 1],
        "velocity": [0.0, 0.0, 0.0],
        "density": 1000.0,
        "color": [50, 100, 200]
    }
    ]

 
    return emitJson2Part(emitjson),range_





def get_partemit_streammultiobjs(scale=1):
        emitjson=[	{
			"objectId": 0,
			"squareCenter":[0.058,1.05,1.0],
			"squareSize": [0.0, 0.4, 1.85],
			"velocity": [1.5, 0.0, 0.0],
			"density": 1000.0,
			"color": [50, 100, 200]
		}]
        squareCenter=emitjson[emitid]["squareCenter"]
        squareCenter=np.array(squareCenter)
        squareCenter*=scale

        temppoints=generate_cube_points(
            center=squareCenter,
            length=emitjson[emitid]["squareSize"][0]*scale,
            height=emitjson[emitid]["squareSize"][2]*scale,
            width= emitjson[emitid]["squareSize"][1]*scale,
            spacing=0.05
        )
    
        points_emit=np.concatenate([points_emit,temppoints])
        tempvel=np.ones_like(temppoints) * emitjson[emitid]["velocity"]
        vel_emit=   np.concatenate([vel_emit, tempvel ])
 

        
def get_partemit_rotatingpanel(scale=1):
        #rorating panel
    range_=range(0, 80, 1)
    
    #VV
    velabs=4
    vely=-1.5



    
    emitjson=[
    {
        "squareCenter":[0.875,1.7,0.1],
        "squareSize": [0.5, 0.5, 0],
        "velocity": [0.0, vely, velabs],
        "density": 1000.0,
        "color": [50, 100, 200]
    },
    {
        "objectId": 0,
        "squareCenter":[3.4,1.7,0.875],
        "squareSize": [0.0, 0.5, 0.5],
        "velocity": [-velabs, vely, 0.0],
        "density": 1000.0,
        "color": [50, 100, 200]
    },
    {
        "objectId": 0,
        "squareCenter":[0.1,1.7,2.625],
        "squareSize": [0.0, 0.5, 0.5],
        "velocity": [velabs,vely, 0.0],
        "density": 1000.0,
        "color": [50, 100, 200]
    },
    {
        "objectId": 0,
        "squareCenter":[2.625,1.7,3.4],
        "squareSize": [0.5, 0.5, 0.0],
        "velocity": [0.0, vely, -velabs],
    }
    ]

    points_emit=np.array([[0],[0],[0]]).T
    vel_emit=   np.array([[0],[0],[0]]).T

    for emitid in range(0,4):
        squareCenter=emitjson[emitid]["squareCenter"]
        squareCenter=np.array(squareCenter)
        squareCenter*=scale
        temppoints=generate_cube_points(
            center=squareCenter,
            length=emitjson[emitid]["squareSize"][0]*scale,
            height=emitjson[emitid]["squareSize"][2]*scale,
            width= emitjson[emitid]["squareSize"][1]*scale,
            spacing=0.05
        )
        
        print('---------')
        print(emitid)
        # print(temp.shape)
        points_emit=np.concatenate([points_emit,temppoints])
        tempvel=np.ones_like(temppoints) * emitjson[emitid]["velocity"]
        print(np.mean(tempvel[:,0]))
        print(np.mean(tempvel[:,1]))
        print(np.mean(tempvel[:,2]))

        #broadcast
        vel_emit=   np.concatenate([vel_emit, tempvel ])
        print(points_emit.shape)
        print(vel_emit.shape)

    return points_emit,vel_emit,range_