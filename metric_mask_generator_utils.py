import torch 
import numpy as np

def mask_generator(self, guidance_points_set):
        #This takes the dictionary which contains the guidance points for each class, and generates the set of masks for each class
        
        if self.metric_type == "test":
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