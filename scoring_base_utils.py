import torch 
from monai.metrics.utils import do_metric_reduction

class ScoreUtils:
    '''
    Assumption is that the class integer codes are in order, and starting from 0, with increments of 1 per class. Class code = 0 refers to background ALWAYS.
    '''
    def __init__(self, score_base, include_background, ignore_empty, include_per_class_scores, class_integer_codes):
        self.score_base = score_base 
        self.include_background = include_background
        self.ignore_empty = ignore_empty
        self.include_per_class_scores = include_per_class_scores
        self.num_classes = len(list(class_integer_codes.keys())) - 1 
        self.dict_class_codes = class_integer_codes 

        self.supported_bases = ["Dice", "Error Rate"]

        if self.score_base.title() not in self.supported_bases:
            #Basic assumption is numbers and symbols will not be placed in the string, only potentially a string with non-capitalised words.
            raise Exception("Selected metric base is not supported")

        if self.score_base == "Dice":
            self.base_class = DiceScoreUtils

        elif self.score_base == "Error Rate":
            self.base_class = ErrorRateUtils 

    def __call__(self, image_masks, pred, gt):

        assert type(image_masks) == tuple, "Score generation failed as the weightmap masks were not in a tuple format"
        assert type(image_masks[0]) == torch.Tensor, "Score generation failed necause the cross class weightmap was not a torch.Tensor"
        assert type(image_masks[1]) == dict, "Score generation failed because the per-class weightmaps were not presented in a class-separated dict."
        assert type(pred) == torch.Tensor, "Score generation failed as the prediction was not a torch.Tensor"
        assert type(gt) == torch.Tensor, "Score generation failed as the gt was not a torch.Tensor"

        return self.base_class(self.ignore_empty, self.include_background, self.include_per_class_scores, image_masks, pred, gt, self.dict_class_codes) 
    

class DiceScoreUtils:

    def dice_score(self, ignore_empty, include_background, include_per_class_scores, image_masks, pred, gt, dict_class_codes):
        
        #This is multi-class generalisable, it implements the summing across all of the classes prior to computing the Dice Score.        
        
        #We assume that the prediction and gt map are discrete.


        assert type(ignore_empty) == bool, 'Dice score generation failed because ignore_empty was not a bool'
        assert type(include_background) == bool, 'Dice score generation failed because include_background was not a bool'
        assert type(include_per_class_scores) == bool, 'Dice score generation failed because include_per_class_scores parameter was not a bool'
        assert type(image_masks) == tuple, 'Dice score generation failed because image_masks were not presented in a tuple consisting of a cross-class mask and a class separated dict of masks'
        assert type(image_masks[0]) == torch.Tensor, 'Dice score generation failed because the cross-class mask was not a torch.Tensor'
        assert type(image_masks[1]) == dict, 'Dice score generation failed because the per-class mask parameter was not a class-separated dict.'

    
        #For multi-class (or binary class where self-include background is TRUE)

        #We weight this according to the class-separated image_masks.  

        cross_class = self.dice_score_multiclass(ignore_empty, include_background, pred, gt, image_masks[0]) 

        per_class_scores = dict() 
        
        for class_label, class_integer_code in dict_class_codes.items():

            if not include_background:
                if class_label.title() == 'Background':
                    continue 
            

            class_sep_pred = torch.where(pred == class_integer_code, 1, 0)
    
            class_sep_gt = torch.where(gt == class_integer_code, 1, 0)


            per_class_scores[class_label] = self.dice_score_per_class(ignore_empty, class_sep_pred, class_sep_gt, image_masks[1][class_label])
        

        assert type(cross_class) == torch.Tensor and cross_class.size() == torch.Size([1]), 'Generation of dice score failed because the cross-class score was not a torch tensor of size 1'
        return (cross_class, per_class_scores)
        
    def dice_score_per_class(self, ignore_empty, class_sep_pred, class_sep_gt, image_mask):

        y_o = torch.sum(torch.where(class_sep_gt > 0, 1, 0) * image_mask)
        y_hat_o = torch.sum(torch.where(class_sep_pred > 0, 1, 0) * image_mask)
        
        weighted_pred = class_sep_pred * image_mask 
        # weighted_gt = class_sep_gt * image_mask 

        intersection = torch.sum(torch.masked_select(weighted_pred, torch.where(class_sep_pred == class_sep_gt, 1, 0) > 0))

        if y_o > 0:
            return torch.tensor([(2 * intersection)/(y_o + y_hat_o)])
        
        if ignore_empty:
            #If we ignore empty then just return a nan value
            return torch.tensor([float("nan")])
        
        if y_o + y_hat_o <=0:
            #If we do not ignore empties, then return a value of 1 if the denom is <=0 (i.e. when both are empty) to indicate full coverage...
            return torch.tensor([1.0])
        
        #else:
        return torch.tensor([0.0])
    
    def dice_score_multiclass(self, ignore_empty, include_background, pred, gt, image_mask):

        '''
        image mask here is the cross-class image mask.
        '''
        if not include_background:
            #if true then background not included
            y_o = torch.sum(torch.where(gt > 0, 1, 0) * image_mask)
            y_hat_o = torch.sum(torch.where(pred > 0, 1, 0) * image_mask)
        else:
            #when background is included, it includes all voxels..
            y_o = torch.sum(torch.ones_like(gt) * image_mask)
            y_hat_o = torch.sum(torch.ones_like(pred) * image_mask)

        if y_o > 0:
            
            intersection = 0
            
            
            for class_label, class_code in self.dict_class_codes.items():
                
                if not include_background:
                    if class_label.title() == 'Background':
                        continue 
                

                pred_channel = torch.where(pred == class_code, 1, 0) 
                
                gt_channel = torch.where(gt == class_code, 1, 0) 

                #The voxel values have already been weighted by the corresponding values in the image mask, so that when we sum it already contains the weight.

                intersection += torch.sum(torch.masked_select(image_mask, torch.where(pred_channel == gt_channel, 1, 0) > 0 )) #* torch.masked_select(image_mask, gt_channel > 0))
                              

            return torch.tensor([(2.0 * intersection) / (y_o + y_hat_o)])
        
        
        if ignore_empty:
            #If we ignore empty then just return a nan value
            return torch.tensor([float("nan")])
        
        denorm = y_o + y_hat_o
        if denorm <= 0:
            #If we do not ignore empties, then return a value of 1 if the denom is <=0 (i.e. when both are empty) to indicate full coverage...
            return torch.tensor([1.0])
        return torch.tensor([0.0])

    def __call__(self, ignore_empty, include_background, include_per_class_scores, image_mask, pred, gt, dict_class_codes):
        
        
        (overall_score, per_class_scores) = self.dice_score(ignore_empty, include_background, include_per_class_scores, image_mask, pred, gt, dict_class_codes)
        
        return {"overall score":overall_score, "per class scores":per_class_scores}    
    
class ErrorRateUtils:
    

    def error_rate(self, ignore_empty, include_background, include_per_class_scores, image_masks, pred, gt, dict_class_codes):
        
        #This is multi-class generalisable.        
        
        #We assume the prediction and gt are discrete.

        #We assume that the image mask contains the information about which voxels are being used for computing the error rate (it may also captures a weight map).
        assert type(ignore_empty) == bool, 'Error rate generation failed because ignore_empty was not a bool'
        assert type(include_background) == bool, 'Error rate generation failed because include_background was not a bool'
        assert type(include_per_class_scores) == bool, 'Error rate generation failed because include_per_class_scores parameter was not a bool'
        assert type(image_masks) == tuple, 'Error rate generation failed because image_masks were not presented in a tuple consisting of a cross-class mask and a class separated dict of masks'
        assert type(image_masks[0]) == torch.Tensor, 'Error rate generation failed because the cross-class mask was not a torch.Tensor'
        assert type(image_masks[1]) == dict, 'Error rate generation failed because the per-class mask parameter was not a class-separated dict.'

    
        class_separated_pred = dict() 

        class_separated_gt = dict()
        
        weighted_errors = 0
        weighted_denom = 0
        per_class_scores = dict()

        for class_label, class_code in dict_class_codes.items():

            if not include_background:
                if class_label.title() == "Background":
                    continue 

            class_separated_pred = torch.where(pred == class_code, 1, 0)
            class_separated_gt = torch.where(gt == class_code, 1, 0)


            (per_class_weighted_errors, per_class_weighted_denom, per_class_error_rate) = self.error_rate_per_class(ignore_empty, class_separated_pred, class_separated_gt, image_masks[1][class_label])

            weighted_errors += per_class_weighted_errors
            weighted_denom += per_class_weighted_denom
            
            if include_per_class_scores:

                per_class_scores[class_label] = per_class_error_rate


        error_rate_cross_class = self.error_rate_comp(weighted_errors, weighted_denom)

        
        return (error_rate_cross_class, per_class_scores)
        
    def error_rate_per_class(self, ignore_empty, class_separated_pred, class_separated_gt, image_mask):
        #For a weighted error rate, given that class segments may have different sizes, we may want to examine class-by-class. 

        #For the denominator, we elect to use the voxels that belong to the ground truth of that class, so that the error rates are balanced according to the size of their own segments.

        disjoint = torch.ones_like(class_separated_pred) - class_separated_pred * class_separated_gt 

        #applying the image mask weightings to these error voxels

        weighted_errors = torch.sum(disjoint * image_mask)

        #computing the denominator (the weighting of the gt voxels) from the gt mask. 

        weighted_denom = torch.sum(class_separated_gt * image_mask) 


        error_rate = self.error_rate_comp(ignore_empty, weighted_errors, weighted_denom)

        return (weighted_errors, weighted_denom, error_rate)
    
    def error_rate_comp(self, ignore_empty, weighted_errors, weighted_denom):

        if weighted_denom > 0:
            #In this case, there were some voxels for this class which had been modified.
            return torch.tensor([weighted_errors/weighted_denom])

        if ignore_empty:
            return torch.tensor([float("nan")])
            
        if weighted_denom <= 0:
            #If there are no changed voxels (this is when the denom would = 0), then just return an error rate of 0.
            return torch.tensor([float(0)])


    def __call__(self, ignore_empty, include_background, include_per_class_scores, image_mask, pred, gt, dict_class_codes):
        

        # (overall_score, per_class_scores) = self.error_rate(ignore_empty, image_mask, pred, gt, num_classes)

        (overall_score, per_class_scores) = self.error_rate(ignore_empty, include_background, include_per_class_scores, image_mask, pred, gt, dict_class_codes)
        
        return {"overall score":overall_score, "per class scores":per_class_scores} 