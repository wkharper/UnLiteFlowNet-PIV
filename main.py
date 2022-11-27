# -*- coding: utf-8 -*-
"""
UnLiteFlowNet-PIV

"""
import argparse
import imageio
from progress.bar import Bar
from src.model.models import *
from src.data_processing.read_data import *
from src.train.train_functions import *
from src.inference.inference import *

data_path = "./sample_data"
result_path = "./output"

def test_train():
    # Read data
    img1_name_list, img2_name_list, gt_name_list = read_all(data_path)
    flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir = read_by_type(
        data_path)

    #img1_len = [len(f_dir) for f_dir in flow_img1_name_list]
    #img2_len = [len(f_dir) for f_dir in flow_img2_name_list]
    #gt_len = [len(f_dir) for f_dir in flow_gt_name_list]

    #for img1_num, img2_num in zip(img1_len, img2_len):
    #    assert img1_num == img2_num
    #for img1_num, gt_num in zip(img1_len, gt_len):
    #    assert img1_num == gt_num

    train_dataset, validate_dataset, test_dataset = construct_dataset(
        img1_name_list, img2_name_list, gt_name_list)

    # Set hyperparameters
    lr = 1e-4
    batch_size = 16
    test_batch_size = 16
    n_epochs = 100
    new_train = True

    # Load the network model
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-5,
                                 eps=1e-3,
                                 amsgrad=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if new_train:
        # New train
        model_trained = train_model(model, train_dataset, validate_dataset,
                                    test_dataset, batch_size, test_batch_size,
                                    lr, n_epochs, optimizer)
    else:
        model_save_name = 'UnsupervisedLiteFlowNet_pretrained.pt'
        PATH = F"./models/{model_save_name}"
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model_trained = train_model(model,
                                    train_dataset,
                                    validate_dataset,
                                    test_dataset,
                                    batch_size,
                                    test_batch_size,
                                    lr,
                                    n_epochs,
                                    optimizer,
                                    epoch_trained=epoch + 1)
    return model_trained


def test_estimate(flow_type,fps,arrow_density):
    flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir = read_by_type(
        data_path)
    assert len(flow_dir) == len(flow_img1_name_list)
    flow_dataset = {}

    for i, f_name in enumerate(flow_dir):
        total_index = np.arange(0, len(flow_img1_name_list[i]), 1)
        flow_dataset[f_name] = FlowDataset(
            total_index, [flow_img1_name_list[i], flow_img2_name_list[i]],
            targets_index_list=total_index,
            targets=flow_gt_name_list[i])

    print("Flow cases: ", flow_type)

    # Visualize results, random select a flow type
    #f_type = random.randint(0, len(flow_type) - 1)
    print("Selected flow scenario: ", flow_type)
    test_dataset = flow_dataset[flow_type]
    test_dataset.eval()
    # ------------Unliteflownet estimation-----------
    number_total = len(test_dataset)
    bar = Bar('Processing', max=number_total)
    unliteflownet = initializeNN()
    error_arr = []
    for number in range(0,number_total):
        input_data, label_data = test_dataset[number]
        runInference(unliteflownet=unliteflownet, input_data=input_data,label_data=label_data,error_arr=error_arr,
                    arrow_density=arrow_density, number=number,resize=True,save_to_disk=True, show=False)
        bar.next()
    bar.finish()

    # Calculate Total Mean Error
    mean_error = np.array(error_arr).mean()
    median_error = np.median(np.array(error_arr))
    std_error = np.std(np.array(error_arr))
    print("Mean Error of dataset = " + str(mean_error) + " px")
    print("Median Error of dataset = " + str(median_error) + " px")
    print("Standard Error of dataset = " + str(std_error) + " px")
    np.savetxt("./output/stats.txt",[mean_error,median_error,std_error])

    print("Done")

    # Create Video
    isVideo = True

    images = []
    if isVideo:
        print("Collecting Image Flows...")
        images = []
        for f in  sorted(glob.iglob(f'{result_path}/*.png')):
            images.append(imageio.imread(f))

        print("Saving Video with " + str(len(images)) + " images...This may take a while. Please wait.")
        imageio.mimsave(result_path + '/movie.gif', images, format='GIF', fps=fps)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='train the model')
    parser.add_argument('--flow' , help='train the model')
    parser.add_argument('--fps' , help='train the model')
    parser.add_argument('--arrow' , help='train the model')

    args = parser.parse_args()
    isTrain = args.train
    isTest = args.test
    flow_type = args.flow
    fps = args.fps
    arrow = args.arrow

    if isTrain:
        test_train()
    if isTest:
        if fps is None:
            fps = 10 # Default
            print("Using default FPS of " + str(fps))
        else:
            print("User selected FPS of " + str(fps))

        if arrow is None:
            arrow = 8 # Default
            print("Using default arrow density of " + str(arrow))
        else:
            print("User selected arrow density of " + str(arrow))
        
        test_estimate(flow_type=flow_type,fps=fps,arrow_density=int(arrow))
