import numpy as np

def gseq(self, arraysize):
    
        n=(arraysize+1)/2;
        sequence=np.zeros((2,arraysize**2))
        sequence[0,0]=n;
        sequence[1,0]=n;			
        dx=+1;
        dy=-1;
        stepx=+1;
        stepy=-1;
        direction=+1;
        counter=0;
      
        for i in range (1,arraysize**2):
            counter=counter+1;
            if (direction==+1):
                sequence[0,i]=sequence[0,i-1]+dx;
                sequence[1,i]=sequence[1,i-1];
                if (counter==abs(stepx)):
                    counter=0;
                    direction=direction*-1;
                    dx=dx*-1;
                    stepx=stepx*-1;
                    if stepx>0:
                        stepx=stepx+1;
                    else:
                        stepx=stepx-1;
            else:
                sequence[0,i]=sequence[0,i-1];
                sequence[1,i]=sequence[1,i-1]+dy;
                if (counter==abs(stepy)):
                    counter=0;
                    direction=direction*-1;
                    dy=dy*-1;
                    stepy=stepy*-1;
                    if (stepy>0):
                        stepy=stepy+1;
                    else:
                        stepy=stepy-1;
        
        seq=(sequence[0,:]-1)*arraysize+sequence[1,:]
        return seq