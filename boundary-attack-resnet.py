#from _future_ import print_function
#try:
#	raw_input
#except:
#	raw_input = input

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec # ??? purpose
import tensorflow as tf
from tensorfllow.keras.datasets import mnist
import pickle # 序列化对象并保存到磁盘中
import time
import datetime
import os
from PIL import Image # Python image library 记得安装
import json
import argparse

#from setup_inception import ImageNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


#???
def orthogonal_perturbation(delta, prev_sample, target_sample):
	prev_sample = prev_sample.reshape(224, 224, 3)
	# Generate perturbation
	perturb = np.random.randn(224, 224, 3)
	perturb /= get_diff(perturb, np.zeros_like(perturb))
	perturb *= delta * np.mean(get_diff(target_sample, prev_sample))
	# Project perturbation onto sphere around target
	diff = (target_sample - prev_sample).astype(np.float32)
	diff /= get_diff(target_sample, prev_sample)
	diff = diff.reshape(3, 224, 224)
	perturb = perturb.reshape(3, 224, 224)
	for i, channel in enumerate(diff):
		perturb[i] -= np.dot(perturb[i], channel) * channel
	# Check overflow and underflow
	mean = [103.939, 116.779, 123.68]
	perturb = perturb.reshape(224, 224, 3)
	overflow = (prev_sample + perturb) - np.concatenate((np.ones((224, 224, 1)) * (255. - mean[0]), np.ones((224, 224, 1)) * (255. - mean[1]), np.ones((224, 224, 1)) * (255. - mean[2])), axis=2)
	overflow = overflow.reshape(224, 224, 3)
	perturb -= overflow * (overflow > 0)
	underflow = np.concatenate((np.ones((224, 224, 1)) * (0. - mean[0]), np.ones((224, 224, 1)) * (0. - mean[1]), np.ones((224, 224, 1)) * (0. - mean[2])), axis=2) - (prev_sample + perturb)
	underflow = underflow.reshape(224, 224, 3)
	perturb += underflow * (underflow > 0)
	return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
	perturb = (target_sample - prev_sample).astype(np.float32)
	perturb /= get_diff(target_sample, prev_sample) # 两个array相除???
	perturb *= epsilon
	return perturb

def get_converted_prediction(sample, classifier):
	sample = sample.reshape(224, 224, 3)
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] += mean[0]
	sample[..., 1] += mean[1]
	sample[..., 2] += mean[2]
	sample = sample[..., ::-1].astype(np.uint8)
	sample = sample.astype(np.float32).reshape(1, 224, 224, 3)
	sample = sample[..., ::-1]
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] -= mean[0]
	sample[..., 1] -= mean[1]
	sample[..., 2] -= mean[2]
	label = decode_predictions(classifier.predict(sample), top=1)[0][0][1]
	return label

def draw(sample, classifier, folder):
	label = get_converted_prediction(np.copy(sample), classifier)
	sample = sample.reshape(224, 224, 3)
	# Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
	# ???
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] += mean[0]
	sample[..., 1] += mean[1]
	sample[..., 2] += mean[2]
	sample = sample[..., ::-1].astype(np.uint8) # the purpose of mean, reverse, conversion
	# Convert array to image and save
	sample = Image.fromarray(sample)
	id_no = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
	# Save with predicted label for image (may not be adversarial due to uint8 conversion)
	sample.save(os.path.join("images", folder, "{}_{}.png".format(id_no, label)))

# why do these preprocessing steps???
def preprocess(ff):
	sample_path = "../imagenetdata/imgs/" + ff
	img = image.load_img(sample_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	x = x.reshape(224, 224, 3)
	if x.shape != (224, 224, 3):
		return None
	return [x, int(ff.split(".")[0])]
	# dimensions of x???

def get_diff(sample_1, sample_2):
	#purpose of reshaping
	sample_1 = sample_1.reshape(3, 224, 224)
	sample_2 = sample_2.reshape(3, 224, 224)
	diff = []
	for i, channel in enumerate(sample_1):
		diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32))) #求范数
	return np.array(diff) #三个维度的L2-norm

def boundary_attack():
	classifier = ResNet50(weights='imagenet')
	# targeted attack
	initial_sample = preprocess('images/original/awkward_moment_seal.png') #adversarial example
	target_sample = preprocess('images/original/bad_joke_eel.png') #original image 
	folder = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
	os.mkdir(os.path.join("images", folder))
	draw(np.copy(initial_sample), classifier, folder) # purpose???
	attack_class = np.argmax(classifier.predict(initial_sample))  #seal
	target_class = np.argmax(classifier.predict(target_sample)) #eel 变成eel样但是label是seal (sample_2_label150_sealion0)

	adversarial_sample = initial_sample
	n_steps = 0
	n_calls = 0
	epsilon = 1.0
	delta = 0.1

	# Move first step to the boundary
	while True:
		trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
		prediction = classifier.predict(trial_sample.reshape(1, 224, 224, 3))
		n_calls += 1
		if np.argmax(prediction) == attack_class:
			adversarial_sample = trial_sample
			break
		else:
			epsilon *= 0.9 # gradually move to the boundary between adversarial and non-adversarial
	

	while True:
		print("Step #{}...".format(n_steps))
		print("\tDelta step...")
		d_step = 0
		while True:
			d_step += 1
			print("\t#{}".format(d_step))
			trial_samples = []
			for i in np.arange(10):
				trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
				trial_samples.append(trial_sample)
			predictions = classifier.predict(np.array(trial_samples).reshape(-1, 224, 224, 3))
			n_calls += 10
			predictions = np.argmax(predictions, axis=1)
			d_score = np.mean(predictions == attack_class)
			if d_score > 0.0:
				if d_score < 0.3:
					delta *= 0.9
				elif d_score > 0.7:
					delta /= 0.9
				adversarial_sample = np.array(trial_samples)[np.where(predictions == attack_class)[0][0]]
				break
			else:
				delta *= 0.9
		print("\tEpsilon step...")
		e_step = 0
		while True:
			e_step += 1
			print("\t#{}".format(e_step))
			trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
			prediction = classifier.predict(trial_sample.reshape(1, 224, 224, 3))
			n_calls += 1
			if np.argmax(prediction) == attack_class:
				adversarial_sample = trial_sample
				epsilon /= 0.5
				break
			elif e_step > 500:
					break
			else:
				epsilon *= 0.5
		n_steps += 1
		chkpts = [1, 5, 10, 50, 100, 500, 1000]
		if (n_steps in chkpts) or (n_steps % 500 == 0):
			print("{} steps".format(n_steps))
			draw(np.copy(adversarial_sample), classifier, folder)
		diff = np.mean(get_diff(adversarial_sample, target_sample))
		if diff <= 1e-3 or e_step > 500:
			print("{} steps".format(n_steps))
			print("Mean Squared Error: {}".format(diff))
			draw(np.copy(adversarial_sample), classifier, folder)
			break
		print("Mean Squared Error: {}".format(diff))
		print("Calls: {}".format(n_calls))
		print("Attack Class: {}".format(attack_class))
		print("Target Class: {}".format(target_class))
		print("Adversarial Class: {}".format(np.argmax(prediction)))

def generate_data(data, model, samples, targeted=False, start=0, seed=3, resnet=False):

	random.seed(seed)
	np.random.seed(seed)
	originals = []
	adversarials = []
	ori_labels = []
	adv_labels = []
	true_ids = []

	for i in range(samples):

		original = data.test_data[start+i]
		ori_label = data.test_labels[start+i]

		if targeted:

			seq = np.random.randint(data.test_labels.shape[0])
			adversarial = data.test_data[seq]
			if resnet:
				while True:
					if seq != start+i:
						break
					seq = np.random.randint(data.test_labels.shape[0])
				adversarial = data.test_data[seq]
			else:
				while True:
					pred_r = model.model.predict(adversarial[np.newaxis, :, :, :])
					if (np.argmax(pred_r, 1) != np.argmax(ori_label)):
						break
					seq = np.random.randint(data.test_labels.shape[0])
					adversarial = data.test_data[seq]
				adv_label=np.eye(data.test_labels.shape[1][np.argmax(pred_r, 1)])

		else:
			adversarial = np.random.random(original.shape)
			if resnet:
				pass
			else:
				
			while True:
				pred_r = model.model.predict(adversarial[None, :, :, :])
				if (np.argmax(pred_r, 1) != np.argmax(ori_label)):
					break
				adversarial = np.random.random(original.shape)
			adv_label = np.eye(data.test_labels.shape[1])[np.argmax(pre_r), 1]

class ImageNet:
	def __init__(self):
		from multiprocessing import Pool
		pool = Pool(8)
		file_list = sorted(os.listdir("../imagenetdata/imgs/"))
		random.shuffle(file_list)
		r = pool.map(preprocess, file_list[:200])
		print(file_list[:200])
		r = [x for x in r if x != None]
		test_data, test_labels = zip(*r)
		self.test_data = np.array(test_data)
		self.test_labels = np.zeros((len(test_labels), 1001))
		self.test_labels[np.arange(len(test_labels)), test_labels] = 1


def main(args):
	
	print("Loading model", args["dataset"])
	if args['dataset'] == 'mnist':
		data, model = MNIST(), MNISTModel("models/mnist.h5")
	elif args['dataset'] == 'cifar10':
		data, model = CIFAR(), CIFARModel("models/cifar.h5")
	elif args['dataset'] = 'imagenet':
		data, model = ImageNet(), ResNet50(weights='imagenet')
	print("Done...")
	
	if args["numing"] == 0:
		args["numing"] = len(data.test_labels) - args['firstimg']
	if args["numing"] > 200 and args["dataset"] == "imagenet":
		args["numing"] = 200
	print("Using", args["numing"], "test images")

	random.seed(args['seed'])
	np.random.seed(args['seed'])
	print("Generate data")
	all_adv, all_ori, all_ori_labels, all_adv_labels, all_true_ids = generate_data(data, model, sample=args['numing'], targeted=args["targeted"],
		start=args["firstimg"], seed = args["seed"], resnet=args["resnet"])
	print("Done...")
	os.system("mkdir -p {}/{}/{}".format(args['save'], args['dataset'], args['attack']))
	img_no = 0
	mse_total = .0
	for i in range(all_true_ids.size):
		original = all_ori[i:i+1]
		adversarial = all_adv[i:i+1]
		ori_label = all_ori_labels[i:i+1]
		adv_label = all_adv_labels[i:i+1]
		true_label = argmax(ori_label)
		print("true labels:", true_label, ori_label)
		original_predict = model.model.predict(original)
		original_predict = np.squeeze(original_predict)
		original_class = np.argsort(original_predict)
		predicted_class = original_class[-1]
		print("original classification:", predicted_class)
		if true_label != predicted_class:
			print("skip wrongly classified image no. {}, original class {}, classified as {}".format(all_true_ids[i], true_label, predicted_class))
			continue

		img_no += 1
		timestart = time.time()
		print("START", img_no, "th ATTACK:")
		adv = boundary_attack(args, model, adversarial, original, ori_label,  adv_label, seed=args["seed"])
		timeend = time.time()
		if len(adv.shape) == 3:
			adv = adv.reshape((1,) + adv.shape)
		mse = np.sum((adv-original)**2)/(adv.shape[0]*adv.shape[1])
		mse_total += mse
		adversarial_predict = model.model.predict(adv)
		adversarial_predict = np.squeeze(adversarial_predict)
		adversarial_class = np.argsort(adversarial_predict)
		adversarial_class = adversarial_class[-1]
		print("adversarial classification:", adversarial_class)
		suffix = "id{}_prev{}_adv{}_dist(MSE){}".format(all_true_ids[i], predicted_class, adversarial_class, mse)
		print("Saving to", suffix)
		show(original, "{}/{}/{}/{}_original_{}.png".format(args['save'], args["dataset"], args["attack"], img_no, suffix))
		show(adv, "{}/{}/{}/{}_adversarial_{}.png".format(args['save'], args["dataset"], args["attack"], img_no, suffix))
		show(adv - original, "{}/{}/{}/{}_diff_{}.png".format(args['save'], args['dataset'], args['attack'], img_no, suffix))
		print("[STATS][MEAN_MSE] total = {}, id = {}, time = {:.3f}, prev_class = {}, new_class = {}, distortion(MSE) = {:.5f}, average MSE: {:.5f}".format(img_no, all_true_ids[i], timeend - timestart, predicted_class, adversarial_class, MSE, MSE_total/img_no))
		sys.stdout.flush()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataset", choices=["mnist, cifar10, imagenet"], default="mnist")
	parser.add_argument("-s", "--save", default="./saved_results_boundary")
	parser.add_argument("-n", "--numing", type=int, default=0, help="number of test images to attack")
	parser.add_argument("-m", "--maxiter", type=int, default=0, help="set 0 to default value")
	parser.add_argument("-f", "--firstimg", type=int, default=0, help="number of image to start with")
	parser.add_argument("-ta", "--targeted", default=False, action="store_true")
	parser.add_argument("-sd", "--seed", type=int, default=373, help="seed for generating random number")

	args = vars(parser.parse_args())
	
	args["resnet"] = False

	if args['targeted']:
		args['attack'] = "targeted"
	else:
		args["attack"] = "untargeted" 
	
	if args["maxiter"] == 0:
		if args["dataset"] == "imagenet":
			if args["untargetd"]:
				args["maxiter"] = 
			else:
				args["maxiter"] = 
		if args["dataset"] == "mnist":
			args["maxiter"] = 
		else:
			args["maxiter"] = 

	#if args["dataset"] == "mnist":
	#	args["use_tanh"] = False
	if args["dataset"] == "imagenet":
		args["resnet"] = True

	random.seed(args["seed"])
	np.random.seed(args["seed"])
	print(args)
	main(args)





