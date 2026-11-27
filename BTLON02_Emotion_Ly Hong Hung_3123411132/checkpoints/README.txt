best_model.pth : BEST MODEL (chon theo Val Acc) - Exp2_A4_ResNet34_TL_128, anh 128x128x3
                 Val 70.55% | Test 71.19% | Macro F1 70.83% | Disgust Recall 72.73% | 85.3 MB

Nap mo hinh:
    from src.models_advanced import load_model_from_checkpoint
    model, ck = load_model_from_checkpoint("checkpoints/best_model.pth", device)
    # ck["arch"]="resnet34", ck["img_size"]=128, ck["in_channels"]=3
    # model.last_conv_layer -> Final Conv Filters; model.gradcam_layer -> Grad-CAM
