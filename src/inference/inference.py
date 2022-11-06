import matplotlib.pyplot as plt
from src.model.models import *
from src.data_processing.read_data import *
from src.train.train_functions import *


def initializeNN():
    # Load pretrained model
    model_save_name = 'UnsupervisedLiteFlowNet_pretrained.pt'
    PATH = F"./models/{model_save_name}"
    unliteflownet = Network()
    unliteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
    unliteflownet.eval()
    unliteflownet.to(device)
    print('unliteflownet load successfully.')
    return unliteflownet



def runInference(unliteflownet, input_data, label_data, number, error_arr, arrow_density=8, resize=False, save_to_disk=False, show=True):
    h_origin, w_origin = input_data.shape[-2], input_data.shape[-1]

    if resize:
        input_data = F.interpolate(input_data.view(-1, 2, h_origin, w_origin),
                                   (256, 256),
                                   mode='bilinear',
                                   align_corners=False)
    else:
        input_data = input_data.view(-1, 2, 256, 256)

    h, w = input_data.shape[-2], input_data.shape[-1]
    x1 = input_data[:, 0, ...].view(-1, 1, h, w)
    x2 = input_data[:, 1, ...].view(-1, 1, h, w)

    # Visualization
    fig, axarr = plt.subplots(1, 2, figsize=(256/(2*arrow_density), 256/(2*arrow_density)))

    # Estimate
    b, _, h, w = input_data.size()
    y_pre = estimate(x1.to(device), x2.to(device), unliteflownet, train=False)
    y_pre = F.interpolate(y_pre, (h, w), mode='bilinear', align_corners=False)

    resize_ratio_u = h_origin / h
    resize_ratio_v = w_origin / w

    u = y_pre[0][0].detach() * resize_ratio_u
    v = y_pre[0][1].detach() * resize_ratio_v

    color_data_pre = np.concatenate((u.view(h, w, 1), v.view(h, w, 1)), 2)
    u = u.numpy()
    v = v.numpy()
    flow_field_norm = np.stack((u,v),axis=2)
    # Draw velocity magnitude
    axarr[1].imshow(fz.convert_from_flow(color_data_pre))
    # Control arrow density
    X = np.arange(0, h, arrow_density)
    Y = np.arange(0, w, arrow_density)
    xx, yy = np.meshgrid(X, Y)
    U = u[xx.T, yy.T]
    V = v[xx.T, yy.T]
    # Draw velocity direction
    axarr[1].quiver(yy.T, xx.T, U, -V)
    axarr[1].axis('off')
    color_data_pre_unliteflownet = color_data_pre

    # ---------------Label data------------------
    if label_data is not None:
        u_gt = label_data[0].detach()
        v_gt = label_data[1].detach()

        color_data_label_gt = np.concatenate((u_gt.view(h, w, 1), v_gt.view(h, w, 1)), 2)
        u_gt = u_gt.numpy()
        v_gt = v_gt.numpy()
        flow_field_norm_gt = np.stack((u_gt,v_gt),axis=2)
        # Draw velocity magnitude
        axarr[0].imshow(fz.convert_from_flow(color_data_label_gt))
        # Control arrow density
        X_gt = np.arange(0, h, arrow_density)
        Y_gt = np.arange(0, w, arrow_density)
        xx_gt, yy_gt = np.meshgrid(X_gt, Y_gt)
        U_gt = u_gt[xx_gt.T, yy_gt.T]
        V_gt = v_gt[xx_gt.T, yy_gt.T]

        # Draw velocity direction
        axarr[0].quiver(yy_gt.T, xx_gt.T, U_gt, -V_gt)
        axarr[0].axis('off')
        color_data_pre_label = color_data_pre

        # Calculate GT Error
        uv_error = np.linalg.norm(np.sqrt(((flow_field_norm_gt - flow_field_norm)**2).mean(axis=0)))
        error_arr.append(uv_error)


    if show:
        plt.show()

    if save_to_disk:
        fig.savefig('./output/frame_' + str(number).zfill(4) + '.png', bbox_inches='tight')
        plt.close()
        reshaped_ffn = flow_field_norm.reshape(flow_field_norm.shape[0], -1)
        reshaped_ffn_gt = flow_field_norm.reshape(flow_field_norm_gt.shape[0], -1)
        np.savetxt("./output/uv_" + str(number).zfill(4) + ".txt", reshaped_ffn, '%.4f')
        if label_data is not None:
            np.savetxt("./output/uv_gt_" + str(number).zfill(4) + ".txt", reshaped_ffn_gt, '%.4f')

    

