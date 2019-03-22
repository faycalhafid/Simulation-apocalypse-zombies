from PIL import Image
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from scipy.spatial import distance
import networkx as nx
import scipy.io
"""
class density :
    def __init__(self, image):
        from PIL import Image
        self.image=Image.open(image)
        self.width=self.image.size[0]
        self.height=self.image.size[1]
        self.colors=self.image.getcolors(self.width*self.height)

pop_density=density('population-density-map.bmp')
elevation=density('elevation1x1_new-mer-bleue.bmp')
"""

im = Image.open('elevation1x1_new-mer-bleue.bmp')
#im=Image.open('warpedimage2.bmp')
population= Image.open('population-density-map.bmp')

# This modules gives useful informations
width=im.size[0]
heigth=im.size[1]
colors = im.getcolors(width*heigth)
print('Nb of different colors: %d' % len(colors))
L=189
colours=[c[1] for c in colors]
"""
RGBsorted=list(colours)
RGBsorted.sort()
displRGB=Image.new("RGB",(L,1))
displRGB.putdata(RGBsorted)
displRGB=displRGB.resize((L,100))
HSVsorted=list(colours)
HSVsorted.sort(key=lambda rgb: colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2]))
displHSV=Image.new("RGB",(L,1))
displHSV.putdata(HSVsorted)
displHSV=displHSV.resize((8*L,500))
"""
#displHSV.show()
#traveling salesman :
# Distance matrix
colours_length=len(colours)
A = np.zeros([colours_length,colours_length])
for x in range(0, colours_length):
    for y in range(0, colours_length):
        A[x,y] = distance.euclidean(colours[x],colours[y])


def NN(A, start):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which
                                   # locations have not been visited
    mask[start] = False
    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]
    return path, cost


path, _ = NN(A, 0)
# Final array
colours_nn = []
for i in path:
    colours_nn.append( colours[i] )
sortedColors=list(colours_nn)
misplaced=sortedColors.pop(-2)
sClrs=[]
sClrs.append(sortedColors[0])
sClrs.append(misplaced)
sClrs.extend(sortedColors[1:])
sortedColors=sClrs
sClrs=[]
misplaced=sortedColors.pop(-1)
sClrs.extend(sortedColors[:sortedColors.index((238, 232, 172))])
sClrs.append(misplaced)
sClrs.extend(sortedColors[sortedColors.index((238, 232, 172)):])
sortedColors=sClrs
displNN=Image.new("RGB",(L,1))
displNN.putdata(sortedColors)
displNN=displNN.resize((8*L,500))
#displNN.show() #BEST RESULT
altitudes=[(i*4.810)/(L-1) for i in range(0,L)]
map_color_altitude=dict(zip(sortedColors,altitudes))
map_color_altitude[(0,0,0)]=0
def closest_color(color, colors):
    colors=np.asarray(colors)
    deltas=colors-color
    dist=np.einsum('ij,ij->i',deltas,deltas)
    c=colors[np.argmin(dist)]
    return (c[0],c[1],c[2])
# got help from https://www.alanzucconi.com/2015/09/30/colour-sorting/
#--------------------------------------------------------------------
pop_width=population.size[0]
pop_height=population.size[1]
pop_colors=population.getcolors(pop_width*pop_height)
pColours=[c[1] for c in pop_colors]
pColours.sort()
brestColor=pColours.pop(-2)
rizeColor=pColours.pop(0)
L2=len(pColours)
popClrs=Image.new("RGB",(L2,1))
popClrs.putdata(pColours)
popClrs=popClrs.resize((10*L2,250))
#popClrs.show()
#densities=[(i*3000)/(L2-1) for i in range(-1,L2)]
densities=[(i*3000)/(L2-1) for i in range(L2)]
map_color_density=dict(zip(pColours,densities))
#map_color_density[(16,16,16)]=0
map_color_density[(255,0,0)]=map_color_density[(255,255,255)]
map_color_density[(0,255,0)]=map_color_density[(255,255,255)]
# OR THIS
dens2s=[(i*3000)/(L2-1) for i in range(-1,L2)]
dens2s[0]=0
map_color_density2=dict(zip(pColours,dens2s))
map_color_density2[(255,0,0)]=map_color_density2[(255,255,255)]
map_color_density2[(0,255,0)]=map_color_density2[(255,255,255)]
#--------------------------------------------------------------------
print("colors : done")
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape[0], arr.shape[1]
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
popu=np.array(population)
pixels = [popu[x:x+15,y:y+15] for x in range(0,popu.shape[0],15) for y in range(0,popu.shape[1],15)]
#elev=np.array(Image.open('warpedimage3.bmp'))
#pixels_elev=[elev[x:x+15,y:y+15] for x in range(0,elev.shape[0],15) for y in range(0,elev.shape[1],15)]
print("images croped into cells")
mini_pop=Image.open('mini-population-density-map.bmp')
mini_elev=Image.open('mini-warpedimage2.bmp')
mini_pop=np.array(mini_pop)
mini_elev=np.array(mini_elev)
width, height=mini_pop.shape[0], mini_pop.shape[1]
G=nx.Graph()

for i in range(width) :
    for j in range(height):
        pixel=mini_pop[i,j]
        cell_elevation=map_color_altitude[closest_color(mini_elev[i,j].tolist(),sortedColors)]
        cell_pop_dens=map_color_density2[closest_color(pixel.tolist(),pColours)]
        G.add_node((i,j),pixel=pixel,elev=cell_elevation,pop_dens=int(cell_pop_dens*15*15),zombies=[],color='green')

def elevation_of(tup):
    i,j=tup[0],tup[1]
    return map_color_altitude[closest_color(mini_elev[i,j].tolist(),sortedColors)]

def dens_of(tup):
    i, j = tup[0], tup[1]
    return int(map_color_density2[closest_color(mini_pop[i,j].tolist(),pColours)]*15*15)

def lamda(node1,node2):
    if node1[0]==node2[0] or node1[1]==node2[1]:
        dist=15
    else :
        dist=15*math.sqrt(2)
    angle=abs((180/math.pi)*np.arctan(abs(G._node[node1]['elev']-G._node[node2]['elev'])/dist))
    if angle==0:
        return 1
    if angle>20:
        return 0
    return 1-angle/20

def neighbors(G):
    for nod in G.node:
        row, col=nod[0], nod[1]
        if row>0:
            c2=(row-1,col)
            if dens_of(c2) > 0 :
                G.add_edge(nod,c2,weight=lamda(nod,c2))
            if col>0:
                c1=(row-1,col-1)
                if dens_of(c1)>0:
                    G.add_edge(nod,c1,weight=lamda(nod,c1))
            if col<height-1:
                c3=(row-1,col+1)
                if dens_of(c3)>0:
                    G.add_edge(nod,c3,weight=lamda(nod,c3))

        if row<width-1:
            c7=(row+1,col)
            if dens_of(c7)>0:
                G.add_edge(nod,c7,weight=lamda(nod,c7))
            if col>0:
                c6=(row+1,col-1)
                if dens_of(c6)>0:
                    G.add_edge(nod,c6,weight=lamda(nod,c6))
            if col<height-1:
                c8=(row+1,col+1)
                if dens_of(c8)>0:
                    G.add_edge(nod,c8,weight=lamda(nod,c8))

        if col>0:
            c4=(row,col-1)
            if dens_of(c4)>0:
                G.add_edge(nod,c4,weight=lamda(nod,c4))
        if col<height-1:
            c5=(row,col+1)
            if dens_of(c5)>0:
                G.add_edge(nod,c5,weight=lamda(nod,c5))

neighbors(G)
print("writing in file")
#nx.write_gexf(G,'mapGraph.gexf')
rize=(140,295)
brest=(87,44)

infected_cells=[]

def is_candidate(cell):
    neighs=list(G.neighbors(cell))
    for neigh in neighs :
        if G._node[neigh]['pop_dens'] > 0 and G._node[neigh]['color']=='green':
            return True
    return False

def put_barriers(G,infected):
    page_rank=nx.pagerank(G,weight='weight')
    from operator import itemgetter
    ranks=sorted([[cell, page_rank[cell]] for cell in infected if is_candidate(cell)],key=itemgetter(1),reverse=True)
    cells_to_colorize=[item[0] for item in ranks[:20]]
    for cell in cells_to_colorize:
        G._node[cell]['color']="blue"
        G._node[cell]['zombies']=[]
        infected.remove(cell)

def start_apocalypse(start, G):
    nb_z=G._node[start]['pop_dens']
    G._node[start]['zombies']=[1 for i in range(nb_z)]
    G._node[start]['pop_dens']=0
    G._node[start]['color']='red'
    if start not in infected_cells:
        infected_cells.append(start)

def get_infected(G):
    return [node for node in G.node if len(G._node[node]['zombies'])>0]

def new_day(infected_cells):
    for node in infected_cells:
        zomb=list(G._node[node]['zombies'])
        G._node[node]['zombies']=[age+1 for age in zomb if age<15]
        if len(G._node[node]['zombies'])==0:
            infected_cells.remove(node)

def spread_to_neighbours(node):
    tot_z=len(G._node[node]['zombies'])
    neighs=list(G.neighbors(node))
    humans_around=sum([G._node[neigh]['pop_dens'] for neigh in neighs])
    dic={}
    for neigh in neighs:
        if G._node[neigh]['color']=="green" or G._node[neigh]['color']=="red":
            neigh_pop=G._node[neigh]['pop_dens']
            if(neigh_pop>0):
                dic[neigh]=math.floor(lamda(neigh,node)*tot_z*(neigh_pop/humans_around))
            else :
                dic[neigh]=0
    for neigh,nb_z in dic.items():
        if nb_z>0:
            random.shuffle(G._node[node]['zombies'])
            zombies_to_go=list(G._node[node]['zombies'][:nb_z])
            G._node[node]['zombies']=G._node[node]['zombies'][nb_z:]
            G._node[neigh]['zombies'].extend(zombies_to_go)
            if neigh not in infected_cells:
                infected_cells.append(neigh)
                G._node[neigh]['color']='red'

def kill_humans(node):
    tot_z=len(G._node[node]['zombies'])
    tot_h=int(G._node[node]['pop_dens'])
    G._node[node]['pop_dens']-=min(tot_h,10*tot_z)
    G._node[node]['zombies'].extend([1 for i in range(min(tot_h,10*tot_z))])

def kill_zombies(node):
    tot_z = len(G._node[node]['zombies'])
    tot_h = int(G._node[node]['pop_dens'])
    new_z=tot_z-min(tot_z,10*tot_h)
    if new_z :
        random.shuffle(G._node[node]['zombies'])
        G._node[node]['zombies']=G._node[node]['zombies'][:new_z]
    else :
        G._node[node]['zombies']=[]
        infected_cells.remove(node)

def draw_state(day,infected):
    mini_pop2=np.copy(mini_pop)
    for pix in infected:
        i,j=pix[0],pix[1]
        old_color=mini_pop2[i,j]
        mini_pop2[i,j]=[220,int(old_color[1]*0.3),int(old_color[2]*0.3)]
    mini_pop3=np.copy(mini_pop)
    for i in range(width):
        for j in range(height):
            old_color=mini_pop3[i,j]
            if G._node[(i,j)]['color']=='red' and (i,j) in infected:
                mini_pop3[i, j] = [220, int(old_color[1] * 0.3), int(old_color[2] * 0.3)]
            if G._node[(i,j)]['color']=='blue':
                mini_pop3[i, j] = [ 0, 0, 220]
            if G._node[(i,j)]['color']=='yellow':
                mini_pop3[i, j] = [220, 220, 0]
    image=Image.fromarray(mini_pop3)
    name="q1_day"+str(day)+".bmp"
    image.save(name)

start_apocalypse(rize,G)
day=1
infected_history={}
nb_zombies={}
day_of_arrival_to_brest=0
arrived=False
while (brest not in infected_cells) and (day<400) and len(infected_cells):
    if brest in infected_cells and not arrived :
        day_of_arrival_to_brest=day
        arrived=True
    temp=list(infected_cells)
    new_day(temp)
    infected_cells=[node for node in temp if len(G._node[node]['zombies']) >0]
    if day==60:
        put_barriers(G,infected_cells)
    infected_history[day]=infected_cells
    draw_state(day,infected_cells)
    print("Day ",day," Number of infected cells : ",len(infected_cells))
    print("Total number of zombies : ", sum([len(G._node[node]['zombies']) for node in infected_cells]))
    nb_zombies[day]=sum([len(G._node[node]['zombies']) for node in infected_cells])
    if day%100==0:
        name1="q1_nb_zombies_day"+str(day)
        scipy.io.savemat(name1,{'nb_z':nb_zombies})
        name2="q1_infected_day"+str(day)
        scipy.io.savemat(name2,{'infected':infected_history})
    inf=list(infected_cells)
    for node in inf:
        spread_to_neighbours(node)
    for node in inf:
        kill_humans(node)
    for node in inf :
        kill_zombies(node)
    day+=1
if brest in infected_cells:
    print("We got to Brest")


