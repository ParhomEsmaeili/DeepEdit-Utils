import os
import sys
# file_dir = os.path.join(os.path.expanduser('~'), 'DeepEditPlusPlus Development/DeepEditPlusPlus')
# sys.path.append(file_dir)


class dice_score_tool():
    def __init__(self):
    
        from monai.transforms import (
            LoadImaged,
            EnsureChannelFirstd,
            Orientationd,
            Compose
            )
        # from monailabel.deepeditPlusPlus.transforms import MappingLabelsInDatasetd  
        from monai.metrics import DiceHelper
        from monai.metrics import DiceMetric
        from monai.utils import MetricReduction

        # self.original_dataset_labels = original_dataset_labels
        # self.label_names = label_names
        # self.label_mapping = label_mapping

        self.transforms_list = [
        LoadImaged(keys=("pred", "gt"), reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=("pred", "gt")),
        Orientationd(keys=("pred", "gt"), axcodes="RAS")
        ]  
        # MappingLabelsInDatasetd(keys="gt", original_label_names=self.original_dataset_labels, label_names = self.label_names, label_mapping=self.label_mapping)
        # ]

        self.transforms_composition = Compose(self.transforms_list, map_items = False)

        self.dice_computation_class = DiceHelper(  # type: ignore
                include_background= False,
                sigmoid = False,
                softmax = False, 
                activate = False,
                get_not_nans = False,
                reduction = MetricReduction.NONE,
                ignore_empty = True,
                num_classes = None
        )
        #self.dice_computation_class = DiceMetric()

    def __call__(self, pred_folder_path, gt_folder_path, image_name):

        pred_image_path = os.path.join(pred_folder_path, image_name)
        gt_image_path = os.path.join(gt_folder_path, image_name)

        input_dict = {"pred":pred_image_path, "gt":gt_image_path}
        output_dict = self.transforms_composition(input_dict)


        dice_score = self.dice_computation_class(y_pred=output_dict["pred"].unsqueeze(0), y=output_dict["gt"].unsqueeze(0))
        #print(dice_score[0])
        return float(dice_score[0]) #float(dice_score) #float(dice_score[0])