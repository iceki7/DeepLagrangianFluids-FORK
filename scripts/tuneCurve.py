def x2(tune0,tuneend,step,num_steps):
    

    return ((num_steps-step)**2) * (tune0-tuneend)/(num_steps**2) + tuneend


def x1(tune0,tuneend,step,num_steps):
    

    return ((num_steps-step)**1) * (tune0-tuneend)/(num_steps**1) + tuneend