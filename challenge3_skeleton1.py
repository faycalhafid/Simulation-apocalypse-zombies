from PIL import Image
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from scipy.spatial import distance
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
elev=np.array(Image.open('warpedimage3.bmp'))
pixels_elev=[elev[x:x+15,y:y+15] for x in range(0,elev.shape[0],15) for y in range(0,elev.shape[1],15)]
print("images croped into cells")
class Cell(object):
    #cells=np.empty([322,234],dtype=object)
    #cells={}
    tempcells=[]
    id=0
    infected=[]
    def __init__(self, pixels, pixels_elevation, row, col):
        self.pixels=pixels
        self.tmp=[]
        for i in range(15):
            for j in range(15):
                self.tmp.append((self.pixels[i][j][0],self.pixels[i][j][1],self.pixels[i][j][2]))
        self.pop_dens=math.floor(sum([map_color_density2[p] for p in self.tmp]))
        self.tmp2=[]
        self.zombies=[]
        #for i in range(15):
        #    for j in range(15):
        #        self.tmp2.append((pixels_elevation[i][j][0],pixels_elevation[i][j][1],pixels_elevation[i][j][2]))
        #self.elevation=np.mean([map_color_altitude[closest_color(p,sortedColors)] for p in self.tmp2])
        self.id=Cell.id
        self.row = self.id//234
        self.col = self.id-234*(self.id//234)
        Cell.id+=1
        #Cell.cells[self.row][self.col]=self
        Cell.tempcells.append(self)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def apocalypse(self):
        nb_z=self.pop_dens
        self.zombies=[1 for i in range(nb_z)]
        self.pop_dens=0
        if self not in Cell.infected:
            Cell.infected.append(self)

    def new_day(self):
        zmb=list(self.zombies)
        self.zombies=[age+1 for age in zmb if age<15]

    def spread_to_neighbours(self, neighs):
        tot_z=len(self.zombies)
        humans_neighbourhood=sum([n.pop_dens for n in neighs])
        dic={}
        for neigh in neighs :
            if (humans_neighbourhood != 0):
                dic[neigh.id]=math.floor(tot_z*(neigh.pop_dens/humans_neighbourhood))
            else :
                dic[neigh.id]=0
        for neigh_id, nb_z in dic.items():
            neigh, neigh_i, neigh_j = getCell_from_id(neigh_id)
            random.shuffle(self.zombies)
            zombies_to_go = list(self.zombies[:nb_z])
            self.zombies = self.zombies[nb_z:]
            neigh.zombies.extend(zombies_to_go)
            if neigh not in Cell.infected :
                Cell.infected.append(neigh)

    def kill_humans(self):
        tot_z=len(self.zombies)
        tot_h=self.pop_dens
        self.pop_dens-=min(tot_h,10*tot_z)
        self.zombies.extend([1 for i in range(min(tot_h,10*tot_z))])

    def kill_zombies(self):
        tot_z = len(self.zombies)
        tot_h = self.pop_dens
        new_z=tot_z-min(tot_z,10*tot_h) #new number of zombies
        if new_z :
            random.shuffle(self.zombies)
            self.zombies=self.zombies[:new_z]
        else :
            self.zombies=[]
            Cell.infected.remove(self)

    def show(self):
        print("row %d col %d pop_dens %f"%(self.row, self.col, self.pop_dens))
print("Creating the cells")
for i in range(len(pixels)) :
    if i%1000==0 :
        print(100*i/len(pixels),"% done")
    Cell(pixels[i],pixels_elev[i],0,0)


def find_pix(pix,cells):
    i=-1
    for arr in cells :
        i+=1
        j=-1
        for cell in arr :
            j+=1
            pex=cell.pixels
            tmp=[(pex[i][j][0],pex[i][j][1],pex[i][j][2]) for i in range(15) for j in range(15)]
            if pix in tmp :
                return [cell, i, j]


def get_cells():
    cells=np.array(Cell.tempcells).reshape((322,234))
    return cells
cells=get_cells()

def neighbors(cells):
    max_i, max_j=cells.shape[0], cells.shape[1]
    neighs={}
    i=0
    for c1 in cells :
        for cell in c1 :
            i+=1
            if i%1000==0:
                print(100*i/(max_i*max_j),"% done")
            n=[]
            i_c, j_c=np.where(cells == cell)
            i_c, j_c=i_c[0],j_c[0]
            cell.row, cell.col=i_c, j_c
            #c1,2,3
            if i_c > 0:
                n.append(cells[i_c-1][j_c]) #append c2
                if j_c > 0 :
                    n.append(cells[i_c-1][j_c-1]) #append c1
                if j_c < max_j-1 :
                    n.append(cells[i_c-1][j_c+1]) #append c3
            #c6,7,8
            if i_c < max_i-1:
                n.append(cells[i_c+1][j_c]) #append c7
                if j_c > 0 :
                    n.append(cells[i_c+1][j_c-1]) #append c6
                if j_c < max_j-2 :
                    n.append(cells[i_c+1][j_c+1]) #append c8
            #c4,5
            if j_c > 0 :
                n.append(cells[i_c][j_c-1]) #append c4
            if j_c < max_j-1 :
                n.append(cells[i_c][j_c+1]) #append c5
            neighs[cell.id]=n
    return neighs
print("Mapping the neighbours")
neighbours=neighbors(cells)

def getCell_from_id(id):
    x,y=cells.shape[0],cells.shape[1]
    return [cells[id//y][id-y*(id//y)],id//y,id-y*(id//y)]

rize_cell, rize_i, rize_j=find_pix((0,255,0),cells)
brest_cell, brest_i, brest_j=find_pix((255,0,0),cells)
rize_cell.apocalypse()
day=1
while (brest_cell not in Cell.infected) and (day<80) and len(Cell.infected):
    Cell.infected=[c for c in Cell.infected if len(c.zombies)]
    print("Day : ",day," Number of infected cells : ",len(Cell.infected))
    print("Spreading to neighbours...")
    inf=list(Cell.infected)
    for inf_cell in inf:
        inf_cell.new_day()
        inf_cell.spread_to_neighbours(neighbours[inf_cell.id])
    print("Killing humans")
    for inf_cell in inf :
        inf_cell.kill_humans()
    print("Killing zombies")
    for inf_cell in inf :
        inf_cell.kill_zombies()
    day+=1
if brest_cell in Cell.infected :
    print("day : ",day," we're in Brest killing these hoes :) ")

def change_pixel(pixel):
    r, g, b=pixel[0], pixel[1], pixel[2]
    if g==24 or g==16 :
        return (30,0,0)
    return (230,int(g*0.3),int(b*0.3))

def check(contenu, contenant):
    for el in contenu :
        if el in contenant :
            return True
    return False

def save_image(cells,infected,i):
    from PIL import Image
    population = Image.open('population-density-map.bmp')
    width, height=population.size
    w2, h2=int(width/6),int(height/6)
    #population.resize((w2,h2))
    for cell in infected :
        for i in range(cell.row,min(cell.row+15,width)):
            for j in range(cell.col,min(cell.col+15,height)):
                cur_pix=population.getpixel((i,j))
                population.putpixel((i,j),change_pixel(cur_pix))
    """
    for x in range(w2):
        for y in range(h2):
            cur_pix=population.getpixel((x,y))
            cur_cel=cells[x//15:(x//15)*6][y//15:(y//15)*6]
            if check(cur_cel,infected):
                population.putpixel((x,y),change_pixel(cur_pix))
    """
    filename="day"+str(i)+".jpg"
    population.save(filename)

save_image(cells, cells[120:125][120:125], 1)