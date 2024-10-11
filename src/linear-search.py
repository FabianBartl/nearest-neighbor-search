
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

# for reproducibility
seed = int(time.time())
# seed = 1722264085
np.random.seed(seed)
print("used seed:", seed)


# https://stackoverflow.com/a/42915422/15774644
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    # Annotate the point xyz with text s
    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy = (xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    # add anotation text s to to Axes3d ax
    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)


# datatypes for type hinting (not for use as constructor class)
class Array(np.ndarray): pass
class NestedArray(Array[Array]): pass

class Point(Array): pass
class Axis(Array): pass
class Index(int): pass

class PointsArray(Array[Point]): pass
class IndicesArray(Array[Index]): pass
class AxesArray(Array[Axis]): pass
class DistancesMatrix(NestedArray[np.float64]): pass
class IndicesMatrix(Array[IndicesArray]): pass

# reshape PointsArray to AxesArray and reverse
def points_to_axes(points:PointsArray) -> AxesArray:
    return np.rot90(np.fliplr(points))
def axes_to_points(axes:AxesArray) -> PointsArray:
    return np.fliplr(np.rot90(axes, -1))

def points_indices_to_points(indices:IndicesArray, points:PointsArray) -> PointsArray:
    return np.array([ points[ind] for ind in indices ])

def create_random_3d_points(num:int, min_max_limit:tuple[float,float]) -> PointsArray:
    return np.floor(np.random.uniform(min_max_limit[0], min_max_limit[1], (num,3)))


# get time in s and store in global var
_t0, _t1 = 0, 0
def t0() -> None:
    global _t0
    _t0 = time.time()

def print_needed_time(desc:str) -> None:
    t1 = time.time()
    global _t0, _t1
    _t1 = t1
    print(f"{desc}: {t1-_t0:_.3f}s")


def plot_nn_points(selected_point_index:Point, nn_point_indices:IndicesArray, points:PointsArray) -> None:
    axes = points_to_axes(points)
    selected_point = points[selected_point_index]
    nn_points = points_indices_to_points(nn_point_indices, points)
    nn_axes = points_to_axes(nn_points)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    # color all points
    nn_point_indices_set = set(nn_point_indices)
    colors = [ ("green" if point_ind in nn_point_indices_set else "blue") for point_ind in range(len(points)) ]
    colors[selected_point_index] = "red"
    
    # plot all points
    ax.scatter(axes[0], axes[1], axes[2], c=colors)

    ax.set_xlabel("axis 0 [X]")
    ax.set_ylabel("axis 1 [Y]")
    ax.set_zlabel("axis 2 [Z]")
    # center selected point and keep all neighbors in view
    space_around = exact_distance_between_points(selected_point, nn_points[-1])
    for axis_ind, set_lim_func in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        set_lim_func(selected_point[axis_ind]-space_around, selected_point[axis_ind]+space_around)

    # https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html#sphx-glr-gallery-mplot3d-surface3d-2-py
    # draw sphere around selected point with distance to furthest neighbor as radius
    # to check neighbor correctnes visually
    radius = exact_distance_between_points(selected_point, nn_points[-1])
    center = selected_point
    resolution = 100
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius*np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius*np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius*np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x,y,z, color=(0.9,0.9,0.9, 0.5))
    ax.set_aspect("equal")

    # connect neighbors to selected point
    for nn_ind, nn_point in enumerate(nn_points):
        ax.plot(*[ (selected_point[axis_ind], nn_point[axis_ind]) for axis_ind in range(3) ], c="green")

    # annotate neighbors
    annotate_kwargs = {"fontsize":10, "xytext":(-3,3), "textcoords":"offset points", "ha":"right", "va":"bottom"}
    for nn_ind, nn_point in enumerate(nn_points):
        annotate3D(ax, s=str(nn_ind), xyz=nn_point, **annotate_kwargs)


def relative_distance_between_points(point_A:Point, point_B:Point) -> np.float64:
    if (point_A == point_B).all():
        return 0
    diff = np.subtract(point_A, point_B)
    return np.sum( diff*diff )

def exact_distance_between_points(point_A:Point, point_B:Point) -> np.float64:
    return np.linalg.norm(point_A - point_B)

def sort_distances_matrix(distances_matrix:DistancesMatrix) -> IndicesMatrix:
    # sort row entries by distance
    return np.argsort(distances_matrix, axis=1, kind="heapsort")

def get_k_nn_of_point(point_index:int, k:int, sorted_indices_matrix:IndicesMatrix) -> IndicesArray:
    # return all neighbors if k is -1
    interval_end = None if k==-1 else k+1
    return sorted_indices_matrix[point_index][1:interval_end]

def calculate_exact_distances_matrix(points:PointsArray) -> DistancesMatrix:
    # initialize infinite distances between all points
    num_points = len(points)
    distances_matrix = np.ones((num_points,num_points)) * np.inf
    
    for ind_A, point_A in enumerate(points):
        for ind_B, point_B in enumerate(points):
            distances_matrix[ind_A][ind_B] = exact_distance_between_points(point_A, point_B)

    return distances_matrix

def calculate_relative_distances_matrix(points:PointsArray) -> DistancesMatrix:
    # initialize infinite distances between all points
    num_points = len(points)
    distances_matrix = np.ones((num_points,num_points)) * np.inf
    
    for ind_A, point_A in enumerate(points):
        for ind_B, point_B in enumerate(points):
            distances_matrix[ind_A][ind_B] = relative_distance_between_points(point_A, point_B)

    return distances_matrix


if __name__ == "__main__":
    # create points
    num_points = int(sys.argv[1]) if len(sys.argv)>=2 and sys.argv[1].replace("_", "").isdigit() else 1_000
    t0()
    points = create_random_3d_points(num_points, (-100,100))
    print_needed_time(f"create {num_points:_d} 3d points O(N)=N")
    
    # calculate exact distances between all points and sort them
    t0()
    exact_distances_matrix = calculate_exact_distances_matrix(points)
    print_needed_time("calculate all exact distances between all points O(N)=N^2")
    t0()
    sorted_exact_indices_matrix = sort_distances_matrix(exact_distances_matrix)
    print_needed_time("sort all exact distances O(N)=N^2*log(N)")

    # calculate relative distances between all points and sort them
    t0()
    relative_distances_matrix = calculate_relative_distances_matrix(points)
    print_needed_time("calculate all relative distances between all points O(N)=N^2")
    t0()
    sorted_relative_indices_matrix = sort_distances_matrix(relative_distances_matrix)
    print_needed_time("sort all relative distances O(N)=N^2*log(N)")

    # get k nearest neighbors (nn) of any selected point index s
    s=0; k=10
    t0()
    nn_exact_point_indices = get_k_nn_of_point(s, k, exact_distances_matrix)
    nn_relative_point_indices = get_k_nn_of_point(s, k, relative_distances_matrix)
    print_needed_time(f"get {k} nearest neighbors of point {s} O(N)=1")

    print(nn_exact_point_indices)
    print(nn_relative_point_indices)
    
    # plot points
    t0()
    plot_nn_points(s, nn_exact_point_indices, points)
    print_needed_time("create plot")
    plt.show()



