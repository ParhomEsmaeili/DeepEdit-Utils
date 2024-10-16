import torch 
from monai.metrics.utils import do_metric_reduction

class ScoreUtils:
    def __init__(self, score_base, include_background):
        self.score_base = score_base 
        self.include_background = include_background
        self.supported_bases = ["Dice", "Error Rate"]

        if self.score_base.title() not in self.supported_bases:
            #Basic assumption is numbers and symbols will not be placed in the string, only potentially a string with non-capitalised words.
            raise Exception("Selected metric base is not supported")

    
    def __call__(self):

        return 
    

class DiceScoreUtils:

    def dice_score(self, ignore_empty, include_background, include_per_class_scores, image_mask, pred, gt, num_classes):
        
        #num_classes is inclusive of the background.

        #This is multi-class generalisable, it implements the summing across all of the classes prior to computing the Dice Score.        
        
        #Split the pred and gt by class, we assume that the prediction map is discrete.

        first_ch = 0 if include_background else 1 

        last_ch = num_classes 

        if (last_ch - first_ch) != 0:
            #For multi-class (or binary class where self-include background is TRUE)

            #We weight this according to the image_mask also.. 

            class_sep_pred = [torch.where(pred == i, 1, 0) * image_mask for i in range(first_ch, last_ch)]
            class_sep_gt = [torch.where(gt == i, 1, 0) * image_mask for i in range(first_ch,last_ch)]
        else:
            #For binary class where self-include background is FALSE
            class_sep_pred = [pred * image_mask]
            class_sep_gt = [gt * image_mask]

        cross_class = self.dice_score_multiclass(ignore_empty, first_ch, last_ch, pred, gt, class_sep_pred, class_sep_gt, image_mask) 

        if include_per_class_scores:    
            per_class_scores = []
            for i in range(last_ch - first_ch):
                per_class_scores.append(self.dice_score_per_class(ignore_empty, class_sep_pred[i], class_sep_gt[i], image_mask))

            return (cross_class, per_class_scores)
        else:
            return (cross_class, [])
        
    def dice_score_per_class(self, ignore_empty, class_sep_pred, class_sep_gt, image_mask):

        y_o = torch.sum(torch.where(class_sep_gt > 0, 1, 0) * image_mask)
        y_hat_o = torch.sum(torch.where(class_sep_pred > 0, 1, 0) * image_mask)
        
        
        intersection = torch.sum(torch.masked_select(class_sep_pred, class_sep_gt > 0))

        if y_o > 0:
            return (2 * intersection)/(y_o + y_hat_o)
        
        if ignore_empty:
            #If we ignore empty then just return a nan value
            return torch.tensor(float("nan"))
        
        if y_o + y_hat_o <=0:
            #If we do not ignore empties, then return a value of 1 if the denom is <=0 (i.e. when both are empty) to indicate full coverage...
            return torch.tensor(1.0)
        
        #else:
        return torch.tensor(0.0)
    
    def dice_score_multiclass(self, ignore_empty, first_ch, last_ch, pred, gt, class_sep_pred, class_sep_gt, image_mask):
        if first_ch:
            #if true then background not included
            y_o = torch.sum(torch.where(gt > 0, 1, 0) * image_mask)
            y_hat_o = torch.sum(torch.where(pred > 0, 1, 0) * image_mask)
        else:
            #when background is included, it includes all voxels..
            y_o = torch.sum(torch.ones_like(gt) * image_mask)
            y_hat_o = torch.sum(torch.ones_like(pred) * image_mask)

        if y_o > 0:
            
            intersection = 0
            
            
            for i in range(last_ch - first_ch):
                pred_channel = class_sep_pred[i]
                gt_channel = class_sep_gt[i] 

                #The voxel values have already been weighted by the corresponding values in the image mask.



                intersection += torch.sum(torch.masked_select(pred_channel, gt_channel > 0)) #* torch.masked_select(image_mask, gt_channel > 0))
                              

            return (2.0 * intersection) / (y_o + y_hat_o)
        
        
        if ignore_empty:
            #If we ignore empty then just return a nan value
            return torch.tensor(float("nan"))
        
        denorm = y_o + y_hat_o
        if denorm <= 0:
            #If we do not ignore empties, then return a value of 1 if the denom is <=0 (i.e. when both are empty) to indicate full coverage...
            return torch.tensor(1.0)
        return torch.tensor(0.0)

    def __call__(self, ignore_empty, include_background, include_per_class_scores, image_mask, pred, gt, num_classes):
        
        
        (overall_score, per_class_scores) = self.dice_score(ignore_empty, include_background, include_per_class_scores, image_mask, pred, gt, num_classes)
        
        return {"overall_score":overall_score, "per_class_scores":per_class_scores}    
    
class ErrorRateUtils:
    

    def error_rate(self, ignore_empty, include_per_class_scores, include_background, image_mask, pred, gt, num_classes):
        
        #This is multi-class generalisable.        
        
        #We assume the prediction and gt are discrete.

        #We assume that the image mask contains the information about which voxels are being used for computing the error rate (implicitly if it also captures a weight map).
        
        first_ch = 0 if include_background else 1 

        last_ch = num_classes 

        if (last_ch - first_ch) != 0:
            #For multi-class (or binary class where self-include background is TRUE)

            #We weight this according to the image_mask also.. 

            class_separated_pred = [torch.where(pred == i, 1, 0) for i in range(first_ch, last_ch)]
            class_separated_gt = [torch.where(gt == i, 1, 0) for i in range(first_ch,last_ch)]
        else:
            #For binary class where self-include background is FALSE
            class_separated_pred = [pred]
            class_separated_gt = [gt]

        
        weighted_errors = 0
        weighted_denom = 0
        per_class_scores = []

        for i in range(last_ch - first_ch):

            (per_class_weighted_errors, per_class_weighted_denom, per_class_error_rate) = self.per_class_extraction(ignore_empty, class_separated_pred[i], class_separated_gt[i], image_mask)

            weighted_errors += per_class_weighted_errors
            weighted_denom += per_class_weighted_denom
            per_class_scores.append(per_class_error_rate)


        error_rate_cross_class = error_rate_cross_class(weighted_errors, weighted_denom)

        if include_per_class_scores:
            return (error_rate_cross_class, per_class_scores)
        else:
            return (error_rate_cross_class, [])

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
            return weighted_errors/weighted_denom

        if ignore_empty:
            return torch.tensor(float("nan"))
            
        if weighted_denom <= 0:
            return torch.tensor(float(0))


    def __call__(self, ignore_empty, image_mask, pred, gt, num_classes):
        

        (overall_score, per_class_scores) = self.error_rate(ignore_empty, image_mask, pred, gt, num_classes)


        return {"overall_score":overall_score, "per_class_score":per_class_scores} 