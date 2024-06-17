from ladybug_geometry.triangulation import earcut
import numpy as np
import cv2 as cv
from shapely.geometry import Polygon
import shapely 
import math 


def triangulate(contours,hierarchy,z=0):
    """
    Calculates triangles from opencv contours and hierarchy obtained using the extract_contours function.

    Parameters
    ----------
    contours: list 
        from opencv.
    hierarchy: list 
        from opencv.
    z: int
        height of the contours in worldspace.
        this is added as the third coordinate to the triangles

    Returns
    ----------
    tris: numpy array (n,3,3)
        containing list of xyz coordinates of vertices making up a n triangles
    """

    #Took this from one of the other libraries but i dont remember which one
    def _DFS(polygons, contours, hierarchy, sibling_id, is_outer, siblings):
        while sibling_id != -1:
            contour = contours[sibling_id].squeeze(axis=1)
            if len(contour) >= 3:
                first_child_id = hierarchy[sibling_id][2]
                children = [] if is_outer else None
                _DFS(polygons, contours, hierarchy, first_child_id, not is_outer, children)

                if is_outer:
                    polygon = [contour, children]
                    polygons.append(polygon)
                else:
                    siblings.append(contour)
            
            sibling_id = hierarchy[sibling_id][0]

    tris = []
    polygons = []
    _DFS(polygons, contours, hierarchy[0], 0, True, [])

    #flattening polygon arrays and extracting hole indices.
    for polygon in polygons:
        flattened_polygon = polygon[0]
        hole_indices = []
        if len(polygon[1])>0:
            outer_len = len(polygon[0])
            hole_indices.append(outer_len)
            for i,hole in enumerate(polygon[1]):
                if i < len(polygon[1])-1:
                    hole_indices.append(hole_indices[-1]+len(hole))
                flattened_polygon = np.append(flattened_polygon,hole,axis=0)

        #using ladybug_geometry.triangulation earcut algorithm
        tri_indices = np.array(earcut(flattened_polygon.flatten(), hole_indices=hole_indices, dim=2))
        tri_indices = np.split(tri_indices,len(tri_indices)/3)
        tris.extend(flattened_polygon[tri_indices])
    
    tris= np.array(tris)

    return(np.concatenate([tris,np.expand_dims(np.ones(tris.shape[:-1]),axis=-1)*z],axis=-1))



def triangulate_polygon(polygon):
    def trian(polygon):
        polygon = [np.array(polygon.exterior.coords),[np.array(interior.coords) for interior in polygon.interiors]]
        flattened_polygon = polygon[0]
        hole_indices = []
        if len(polygon[1])>0:
            outer_len = len(polygon[0])
            hole_indices.append(outer_len)
            for i,hole in enumerate(polygon[1]):
                if i < len(polygon[1])-1:
                    hole_indices.append(hole_indices[-1]+len(hole))
                flattened_polygon = np.append(flattened_polygon,hole,axis=0)

        #using ladybug_geometry.triangulation earcut algorithm
        tri_indices = np.array(earcut(flattened_polygon[:,:2].flatten(), hole_indices=hole_indices, dim=2))
        tri_indices = np.split(tri_indices,len(tri_indices)/3)
        tris = flattened_polygon[tri_indices]
        return(tris)
    
    if polygon.geom_type == "MultiPolygon":
        tris = []
        for subpoly in polygon.geoms:
            tris.extend(trian(subpoly))
    else:
        tris = trian(polygon)
    return(tris)


def extract_contours(img,epsilon_factor=0.05):
    """
    Extracts the contours from an image using the open-cv approxPolyDP function

    Parameters
    ----------
    img: np array(n,n,3) 
        array containing color image from which contours should be extracted
    epsilon_factor: float
        controls the refinement of the contour, lower numbers give more accurate representations using more lines  


    Returns
    ----------
    approximations: list
        coordinates of polygon vertices
    hierarchy:      list
        hierarchy of extracted polygons
    """

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(img_gray,127,255,0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #contains the points making up the appoximated polygon
    approximations = []    
    for contour in contours:
        epsilon = epsilon_factor*cv.arcLength(contour,True)
        approx = cv.approxPolyDP(contour,epsilon,True)
        approximations.append(approx)
    return(approximations,hierarchy)


def polygons_from_opencv(contours, hierarchy):
    """
    Calculates polygons from opencv contours and hierarchy obtained using the extract_contours function.

    Parameters
    ----------
    contours: list 
        from opencv.
    hierarchy: list 
        from opencv.

    Returns
    ----------
    polygons: shapely.Multipolygon
        contains a multipolygon approximating the contours extracted by openCV
    """

    def _DFS(polygons, contours, hierarchy, sibling_id, is_outer, siblings):
        while sibling_id != -1:
            contour = contours[sibling_id].squeeze(axis=1)
            if len(contour) >= 3:
                first_child_id = hierarchy[sibling_id][2]
                children = [] if is_outer else None
                _DFS(polygons, contours, hierarchy, first_child_id, not is_outer, children)

                if is_outer:
                    polygon = Polygon(contour, holes=children)
                    polygons.append(polygon)
                else:
                    siblings.append(contour)

            sibling_id = hierarchy[sibling_id][0]

  
    hierarchy = hierarchy[0]
    polygons = []
    _DFS(polygons, contours, hierarchy, 0, True, [])
    polygons = shapely.MultiPolygon(polygons)
    return polygons




# might have to be redone
def substring_from_points(points,boundary_geoms):
    """
    Extracts substring from using points projected onto the closest boundary of boundaries 

    Parameters
    ----------
    points: shapely.MultiPoint 
        shapely multipoints containing beginning and end point of the substring.
    boundary_geoms: polygon.boundary.geoms 
        shapely boundary geoms from polygon which you want to extarct substrings from.

    Returns
    ----------
    Linestring: shapely.LineString
        Linestring which contains segment of the original boundary.
    """

    _centroid = points.centroid
    _closest_dis = 10000000
    _closest_boundary = 10000000
    for _boundary in list(boundary_geoms):
        _distance = _centroid.distance(_boundary)
        if _distance < _closest_dis:
            _closest_dis = _distance
            _closest_boundary = _boundary

    par_on_boundary = [_closest_boundary.line_locate_point(point) for point in points.geoms]
    substrings = []
    for i in range(math.ceil(len(par_on_boundary)/2)):
        substrings.append(shapely.ops.substring(_closest_boundary,par_on_boundary[i*2],par_on_boundary[i*2+1]))
    return(shapely.multilinestrings(substrings))


def substrings_from_parameters(params,boundary_geom):
    if len(params)%2 != 0:
        params.extend([0])
    _segments = np.column_stack([params,np.roll(params,-1,axis=0)])[0:-1:2]
    _substrings = shapely.MultiLineString([shapely.ops.substring(boundary_geom,_segment[0],_segment[1],True) for _segment in _segments])
    return(_substrings)



def image_to_shapely(img,epsilon=.1):
    contours, hierarchy = cv.findContours(np.where(img>.5,1,0).astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    polygons = polygons_from_opencv(contours,hierarchy).simplify(epsilon)
    if polygons.geom_type != "MultiPolygon":
        polygons= shapely.MultiPolygon([polygons])

    
    return(polygons)

def filter_polygons_size(multipolygon,min_size=20,only_largest=False):
    filtered_polygons = []
    largest = 0
    for polygon in multipolygon.geoms:
        if only_largest:
            if polygon.area > min_size and polygon.area > largest:
                largest = polygon.area
                filtered_polygons = [polygon]
        else:
            if polygon.area > min_size:
                filtered_polygons.append(polygon)
    return(shapely.MultiPolygon(filtered_polygons))

def tris_to_obj(tris,filename='./temp.obj'):
    i = 1
    f = open(filename, "w")
    for tri in tris:
        for coord in tri:
            f.write(f'v {coord[1]} {coord[2]} {coord[0]}\n')

    for _ in range(len(tris)):
        f.write(f"f {i} {i+1} {i+2}\n")
        i+=3
    f.write(f'\n')
    f.close()


def draw_polygon(polygon,image):
    if polygon.geom_type== 'MultiPolygon':      
        for subpoly in polygon.geoms:
            cv.fillPoly(image,np.expand_dims(np.array(subpoly.exterior.coords,dtype='int32'),0),1)
            for interior in subpoly.interiors:
                cv.fillPoly(image,np.expand_dims(np.array(interior.coords,dtype='int32'),0),0)
        return(image)
    else:
        cv.fillPoly(image,np.expand_dims(np.array(polygon.exterior.coords,dtype='int32'),0),1)
        for interior in polygon.interiors:
            cv.fillPoly(image,np.expand_dims(np.array(interior.coords,dtype='int32'),0),0)
        return(image)