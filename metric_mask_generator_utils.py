import torch 
import numpy as np
from itertools import chain

class MaskGenerator:
    def __init__(self, click_map_types, gt_map_types, human_measure):
        
        #We use dicts for the following, since each weight-map type may require some parameterisations.
        self.click_weightmap_types = dict()
        for key in click_map_types.keys():
            self.click_weightmap_types[key.title()] = click_map_types[key]     
        
        #A dict of the components of the weight-map which may solely originate from the click information/click set.
        self.gt_weightmap_types = dict()
        for key in gt_map_types.keys():
            self.gt_weightmap_types[key.title()] = gt_map_types[key]
         
        self.gt_weightmap_types = [i.title() for i in list(gt_map_types.keys())] 
        #A dict of the components of the weight-map which may originate from the ground truth in relation to the clicks, e.g. the connected component
        
        self.human_measure = [i.title() for i in human_measure] 
        #The measure of model performance being implemented, e.g. responsiveness in region of locality/non-worsening elsewhere. This is a list of strings only!


        self.supported_click_weightmaps = ['Ellipsoid',
                                            'Cuboid', 
                                            'Scaled Normalised Euclidean Distance',
                                            'Exponentialised Scaled Normalised Euclidean Distance',
                                            '2D Intersections', 
                                            'None']
        
        self.supported_gt_weightmaps = ['Connected Component',
                                        'None']
        
        self.supported_human_measures = ['Local Responsiveness',
                                        'Non Worsening',
                                        'None']

        if any(list(self.click_weightmap_types.keys())) not in self.supported_click_weightmaps:
            #Basic assumption is numbers and symbols will not be placed in the string, only potentially a string with non-capitalised words.
            raise Exception("Selected click-based weight map is not supported")
        
        elif any(list(self.gt_weightmap_types.keys())) not in self.supported_gt_weightmaps:
            raise Exception("Selected gt-based weight map is not supported")
        
        elif any(self.human_measure) not in self.supported_human_measures:
            raise Exception("Selected human-centric measure is not supported")
        



    def click_based_weightmaps(self, guidance_points_set, image_dims):
        #The guidance points set is assumed to be a dictionary covering all classes, with each point being provided as a 2D/3D set of coordinates.
        
        list_of_points = list(chain.from_iterable(list(guidance_points_set.values())))


        for item in self.click_weightmap_types.items():
            
            if item[0] == "Ellipsoid":    
                mask = torch.zeros(self.image_dims)

                mask = self.generate_ellipsoid(list_of_points, item[1], mask)

            elif item[0] == "Cuboid":
                mask = torch.zeros(self.image_dims)
        
                mask = self.generate_cuboid(list_of_points, item[1], mask)

            elif item[0] == "Scaled Normalised Euclidean Distance":
                #A scaled euclidean-distance that is first normalised by the image dimensions in each relative dimension. 

                #For scale factor = 1, this reduces to the original normalised euclidean distance.

                #Requires parametrisation. 

                
                #Create a weight map for each point and fuse together. 

                pass  

            elif item[0] == "Exponentialised Scaled Normalised Euclidean Distance":
                pass 

            elif item[0] == "2D Intersections":
                mask = self.generate_axial_intersections(list_of_points)

            elif item[0] == "None":
                mask = torch.ones(self.image_dims)


            

            
            
        return mask 
        
    def gt_based_weightmaps(self, image_dims):

        return mask 
    
    def human_measure_weightmap(self, click_based_weightmaps, gt_based_weightmaps):
        return mask 
    


    def __call__(self, guidance_points_set, image_dims):
            
            #This takes the dictionary which contains the guidance points for all classes. This set of guidance points will be assumed to be in the same orientation of the images. 
            
            image_dims = image_dims 
            #A list containing the dimensions of the image, for which the weight-maps will be created, assumed to be in RAS orientation for 3D images.


            masks = []  
            masks.append(mask)


            if self.metric_type == "None":
                #we just want to be able to test that our base utility is still working, so we will use a full tensor of ones as the image mask
                mask = torch.ones(self.image_size)

            if self.metric_type == "ellipsoid":

                roi_size = np.round(np.array(self.image_size) * self.metric_click_size) 
                mask = self.mask_apply(guidance_points_set, roi_size, ellipsoid_shape_bool=True)

            #We round in the case that we have a decimal dimension size. The roi_size dimensions are also for only one quadrant size (i.e. half in each direction)

            elif self.metric_type == "cuboid":

                roi_size = np.round(np.array(self.image_size) * self.metric_click_size)
                mask = self.mask_apply(guidance_points_set, roi_size, ellipsoid_shape_bool=False)
            
            elif self.metric_type == "distance":

                mask = self.distance_mask(guidance_points_set)

            elif self.metric_type == "scaled_distance":

                mask = self.scaled_distance_mask(guidance_points_set)
            
            elif self.metric_type == "2d_intersections":

                mask = self.intersection_mask(guidance_points_set)
            

            #Compounding factor, in order to restrict the score computation to a sub-region of the image, e.g. based on whether it should only be computed for the connected component a click is placed
            #in.
            if self.metric_limiter == "connected_component":

                mask_2 = self.connected_component_mask(guidance_points_set)

            else:
                #default is that there is no limiter, and so only the original guidance mask is of concern.
                mask_2 = np.ones(self.image_size)
            


            if self.human_measure == "locality":
                #If locality is the measure then no change required.
                #output_mask_current_class
                output_mask = mask 

            elif self.human_measure == "temporal_consistency":
                #If temporal consistency is the measure, then we need to invert the mask.
                #output_mask_current_class

            
                output_mask = 1 - mask

            #TODO: add possible combinations of the mask, e.g. temporal consistency requires us to look at the changed voxels which we fused with the image-mask weightings.


            # masks[label_class] = output_mask_current_class

            return output_mask


    def generate_axial_intersections(self, list_of_points):
        # Non-parametric, only depends on the axial slices which could've been used to place the click on a 2D monitor. This is only applicable for 3D images!
        
        #Checking that the image dims are indeed 3 dimensional.

        if len(self.image_dims) < 3:
            raise Exception("Selected 2D Intersections for a 2D image!") 
        
        mask = torch.zeros(self.image_dims)
        #We initialise a tensor of zeroes, then we will insert 1s at the axial slices in which clicks could have feasibly been placed! 

        for point in list_of_points:
            mask[point[0],:,:] = 1
            mask[:,point[1],:] = 1
            mask[:,:,point[2]] = 1

        return mask 
    

    def generate_cuboid(self, list_of_points, scale_parametrisation, mask):
        #Cuboid requires parameterisation.

        #Parameterisation is a list of scales relative to the image dimension (e.g., 0.1)

        #This should be used to produce the raw quantity of voxels..(e.g. 50 voxels in x, 75 in y, 90 in z for a 500 x 750 x 900 image)

        if len(scale_parametrisation) == 1:
            parametrisation = torch.round(scale_parametrisation * torch.tensor(mask.size()))
        else:
            parametrisation = [torch.round(scale_parametrisation[i] * torch.tensor(mask.size()[i])) for i in range(len(scale_parametrisation))]

        #We assume that the click is being placed in the center of each voxel.
        
        for point in list_of_points:
            #obtain the extreme points of the cuboid which will be assigned as the box region:
            min_maxes = []
            
            for index, coordinate in enumerate(point):
                #For each coordinate, we obtain the extrema points.
                dimension_min = int(max(0, coordinate - parametrisation[index]))
                dimension_max = int(min(self.image_size[index] - 1, coordinate + parametrisation[index]))

                min_max = [dimension_min, dimension_max] 
                min_maxes.append(min_max)

            if len(self.image_size) == 2:
            #If image is 2D            
                mask[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1]] = 1
            elif len(self.image_size) == 3:
                #If image is 3D:
                mask[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1], min_maxes[2][0]:min_maxes[2][1]] = 1
        
        return mask


    def generate_ellipsoid(self, list_of_points, scale_parametrisation, mask):
        #Ellipsoid requires parametrisation: There are three options available

        #param_1 only: All dimensions have the same scaling, relative to the corresponding image dimension.

        #param_1/2 or param_1/2/3 indicate separate scalings relative to each corresponding image dimension.


        #Ellipsoids is defined in the following manner: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1


        if len(scale_parametrisation) == 1:
            #Ellipsoid is equally scaled relative to the corresponding image dimension.
            
            #We generate each a,b,c by multiplying the scale_parameterisation by the image dimension
            scale_factor_denoms = torch.round(scale_parametrisation * torch.tensor(mask.size()))
        else:
            #It is not scaled equally relative to the corresponding image dimension.
            scale_factor_denoms = [torch.round(scale_parametrisation[i] * torch.tensor(mask.size()[i])) for i in range(len(scale_parametrisation))] 

        for point in list_of_points:
             


    def check_in_ellipsoid(self,centre, scale_factor_denoms, point, image_dims):
        #Checking if a point is inside an ellipsoid defined using the centre, and the scale parameters a,b,c.

        #Ellipsoids is defined in the following manner: (x-xo/a)^2 + (y-yo/b)^2 + (z-zo/c)^2 = 1 (for 2D the z-term is just dropped)

        #We test the validity of a point by treating point coordinates as being at the center of each voxel.  

        centre = [coord + 0.5 if coord < image_dims[i] else coord - 0.5 for i, coord in enumerate(centre)]
        point = [coord + 0.5 if coord < image_dims[i] else coord - 0.5 for i, coord in enumerate(point)]
        
        #finding the difference in each dimension:
        numerators = [centre[i] - point[i] for i in range(len(centre))]
        lhs_comp = [(numerators[i]/scale_factor_denoms[i]) for i in range(len(numerators))].sum()

        #check if the point lies inside the ellipsoid:

        return lhs_comp <= 1