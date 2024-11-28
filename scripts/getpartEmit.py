from getlattice import generate_cube_points
import numpy as np

def emitJson2Part(emitjson,scale):
    squareCenter=emitjson[0]["squareCenter"]
    squareCenter=np.array(squareCenter)
    squareCenter*=scale

    spacing=0.05
    if("spacing" in emitjson[0]):
        spacing=emitjson[0]["spacing"]

    points_emit=generate_cube_points(
        center=squareCenter,
        length=emitjson[0]["squareSize"][0]*scale,
        height=emitjson[0]["squareSize"][2]*scale,
        width= emitjson[0]["squareSize"][1]*scale,
        spacing=spacing
    )
    vel_emit=np.ones_like(points_emit) * emitjson[0]["velocity"]
        # spacing=0.047
    return points_emit,vel_emit

def get_partemit_propellerlargefill(scale=1):
    range_=range(0, 1, 1)

    emitjson=[
    {
    "objectId": 0,
    "squareCenter":[0, -0.65, 0],
    "squareSize": [3.93, 1.63, 3.93],
    "translation": [0.0, 0.0, 0.0],
    "scale": [1, 1, 1],
    "velocity": [0.0, 0.0, 0.0],
    "density": 1000.0,
    "color": [50, 100, 200],
    "spacing":0.05
    }
    ]
    points_emit,vel_emit=emitJson2Part(emitjson,scale)

    return points_emit,vel_emit,range_

def get_partemit_taylorvortex(scale=1):
    range_=range(0, 1, 1)

    emitjson=[
    {
    "objectId": 0,
    "squareCenter":[-0.888, -0.3, 1.20469],
    "squareSize": [4.8, 0.7, 4.8],

    "translation": [0.0, 0.0, 0.0],
    "scale": [1, 1, 1],
    "velocity": [0.0, 0.0, 0.0],
    "density": 1000.0,
    "color": [50, 100, 200]
    }
    ]
    points_emit,vel_emit=emitJson2Part(emitjson,scale)

    return points_emit,vel_emit,range_
def get_partemit_propellerbiglarge2(scale=1):
    range_=range(0, 1, 1)
    


    emitjson=[
    {
    "objectId": 0,
    "squareCenter":[0, 0.111, 0],
    "squareSize": [4.8, 1.5, 4.8],
    "translation": [0.0, 0.0, 0.0],
    "scale": [1, 1, 1],
    "velocity": [0.0, 0.0, 0.0],
    "density": 1000.0,
    "color": [50, 100, 200],
    "spacing":0.049
    }
    ]
    points_emit,vel_emit=emitJson2Part(emitjson,scale)

    return points_emit,vel_emit,range_


def get_partemit_propellerlarge2(scale=1):
    range_=range(0, 1, 1)

    emitjson=[
    {
    "objectId": 0,
    "squareCenter":[0, -0.3, 0],
    "squareSize": [4.8, 0.7, 4.8],

    "translation": [0.0, 0.0, 0.0],
    "scale": [1, 1, 1],
    "velocity": [0.0, 0.0, 0.0],
    "density": 1000.0,
    "color": [50, 100, 200]
    }
    ]
    points_emit,vel_emit=emitJson2Part(emitjson,scale)

    return points_emit,vel_emit,range_

def get_partemit_propellerlarge(scale=1):
    range_=range(0, 1, 1)
   

    emitjson=[
    {
    "objectId": 0,
    "squareCenter":[0, -0.65, 0],
    "squareSize": [3.85, 1.6, 3.85],
    "translation": [0.0, 0.0, 0.0],
    "scale": [1, 1, 1],
    "velocity": [0.0, 0.0, 0.0],
    "density": 1000.0,
    "color": [50, 100, 200]
    }
    ]
    points_emit,vel_emit=emitJson2Part(emitjson,scale)

    return points_emit,vel_emit,range_


def get_partemit_propeller(scale=1):
    range_=range(0, 1, 1)
    emitjson=[
    {
    "objectId": 0,
    "squareCenter":[0, -0.01, 0],
    "squareSize": [1.8, 1.8, 1.8],
    "translation": [0.0, 0.0, 0.0],
    "scale": [1, 1, 1],
    "velocity": [0.0, 0.0, 0.0],
    "density": 1000.0,
    "color": [50, 100, 200]
    }
    ]
    points_emit,vel_emit=emitJson2Part(emitjson,scale)

    return points_emit,vel_emit,range_

def get_partemit_wavetowerstatic(scale=1):
    range_=range(0, 1, 1)


    emitjson=[
    {
    "objectId": 0,
    "squareCenter":[6.01, 0.28, 3],
    "squareSize": [11.85, 0.5, 5.85],
    "translation": [0.0, 0.0, 0.0],
    "scale": [1, 1, 1],
    "velocity": [0.0, 0.0, 0.0],
    "density": 1000.0,
    "color": [50, 100, 200]
    }
    ]
    points_emit,vel_emit=emitJson2Part(emitjson,scale)
    print('actual center')
    print(np.mean(points_emit[:,0]))
    print(np.mean(points_emit[:,1]))
    print(np.mean(points_emit[:,2]))

    return points_emit,vel_emit,range_


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

    points_emit,vel_emit=emitJson2Part(emitjson,scale)
    return points_emit,vel_emit,range_



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

 
    points_emit,vel_emit=emitJson2Part(emitjson,scale)
    return points_emit,vel_emit,range_

# 2x:
# 0.048/0.016=3
def get_partemit_streammultiobjsHorizon(scale=1):
    range_=range(0, 500, 1)
    




    emitjson=[	{
        "objectId": 0,
        "squareCenter":[0.058,0.25,1.0],
        "squareSize": [0.0, 0.4, 1.85],
        "velocity": [2.9, 0, 0],
        "density": 1000.0,
        "color": [50, 100, 200]
    }]



    points_emit,vel_emit=emitJson2Part(emitjson,scale)
    return points_emit,vel_emit,range_


def get_partemit_streammultiobjs(scale=1):
    range_=range(0, 140, 1)
    
       


    emitjson=[	{
        "objectId": 0,
        "squareCenter":[0.058,1.05,1.0],
        "squareSize": [0.0, 0.4, 1.85],
        "velocity": [3.0, -0.5, 0.0],
        "density": 1000.0,
        "color": [50, 100, 200]
    }]

    if(scale==2):
        emitjson[0]["velocity"]=[3.0, -0.5, 0.0]    
    elif(scale==3):
        #stream3xcut
        emitjson[0]["velocity"]=[3.5, -0.5, 0.0]


        #test
        # emitjson[0]["velocity"]=[3.0, -0.5, 0.0]
        #thin
        # emitjson[0] ["squareCenter"][1]=1
        # emitjson[0]["squareSize"]   [1]=0.3        
        # #thick
        # emitjson[0] ["squareCenter"][1]=1.2
        # emitjson[0]["squareSize"]   [1]=0.7
        # emitjson[0]["velocity"]=[3.7, -0.3, 0.0]
        # range_=range(0, 100, 2)
        # emitjson[0] ["squareCenter"][0]+=2
        # emitjson[0] ["squareCenter"][1]+=-0.3
        # emitjson[0]["spacing"]=0.047
        # emitjson[0] ["squareCenter"]=[1.858,1,1.0]
        # emitjson[0]["squareSize"]=[0.0, 0.3, 1.85]
        # range_=range(0, 100, 1)


        #3xcutHorizon
        emitjson[0] ["squareCenter"][0]=3
        emitjson[0] ["squareCenter"][1]=0.25
        emitjson[0]["velocity"][1]=0
        emitjson[0]["squareSize"][1]=0.3
        emitjson[0] ["squareCenter"][1]-=0.05





        



    else:
        assert(False)


    points_emit,vel_emit=emitJson2Part(emitjson,scale)
    return points_emit,vel_emit,range_

        
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