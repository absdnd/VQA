import os
import json 
import pickle 
from collections import Counter
from torchtext import data
from os.path import join


def _create_tsv(data_dir, quesfile, ansfile, outfile, ansid=None):
	quesfile = join(data_dir, quesfile)
	ques_json = json.load(open(quesfile))
	ques = [q['question'] for q in ques_json['questions']]
	quesid = [q['question_id'] for q in ques_json['questions']]
	imgid = [q['image_id'] for q in ques_json['questions']]
	if ansfile is not None:
		ansfile = join(data_dir, ansfile)
		ans_json = json.load(open(ansfile))
		ans = [a['multiple_choice_answer'] for a in ans_json['annotations']]
		k = 1000
		if ansid is None:
				c = Counter(ans)
				topk = c.most_common(n=k)
				ansid = dict((a[0], i) for i, a in enumerate(topk))

				ans_itos_file = join(data_dir, 'ans_itos.tsv')
				print('Dumping ans-to-idx map to {}'.format(ans_itos_file))
				with open(ans_itos_file, 'w') as fout:
					for i, (a, freq) in enumerate(topk):
						fout.write('{}\t{}\t{}'.format(i, a, freq) + '\n')
					fout.write('{}\t{}\t{}'.format(k, '<unk>', 'rest') + '\n')

		ans = [ansid[a] if a in ansid else k for a in ans]
	else:
		ans = [0 for q in ques]
	outfile = join(data_dir, outfile)

	with open(outfile, 'w') as out:
			for q, qid, i, a in zip(ques, quesid, imgid, ans):
					out.write('\t'.join([str(qid), q, str(i), str(a)]) + '\n')


def _create_small_tsv(data_dir, quesfile, ansfile, outfile, dataset_size = 10, ansid = None): 
	
	quesfile = join(data_dir, quesfile)
	ques_json = json.load(open(quesfile))
	ques = [q['question'] for q in ques_json['questions'][:dataset_size]]
	quesid = [q['question_id'] for q in ques_json['questions'][:dataset_size]]
	imgid = [q['image_id'] for q in ques_json['questions'][:dataset_size]]

	if ansfile is not None:
		ansfile = join(data_dir, ansfile)
		ans_json = json.load(open(ansfile))
		ans = [a['multiple_choice_answer'] for a in ans_json['annotations']]
		k = 1000
		if ansid is None:
			c = Counter(ans)
			topk = c.most_common(n=k)
			ansid = dict((a[0], i) for i, a in enumerate(topk))

			ans_itos_file = join(data_dir, 'ans_itos_small.tsv')
			print('Dumping ans-to-idx map to {}'.format(ans_itos_file))
			with open(ans_itos_file, 'w') as fout:
				for i, (a, freq) in enumerate(topk):
					fout.write('{}\t{}\t{}'.format(i, a, freq) + '\n')
				fout.write('{}\t{}\t{}'.format(k, '<unk>', 'rest') + '\n')

		####### Loading the answer file for answer in question #########
		ans = [ansid[a] if a in ansid else k for a in ans]
	else:
		ans = [0 for q in ques]
	outfile = join(data_dir, outfile)

	with open(outfile, 'w') as out:
			for q, qid, i, a in zip(ques, quesid, imgid, ans):
				out.write('\t'.join([str(qid), q, str(i), str(a)]) + '\n')




def _create_loaders(path, traintsv, valtsv):
	def parse_int(tok, *args):
		return int(tok)
	
	quesid = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))
	ques = data.Field(include_lengths=True)
	imgid = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))
	ans = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))
	train_data, val_data = data.TabularDataset.splits(path=path, train=traintsv, validation=valtsv,
													  fields=[('quesid', quesid), ('ques', ques), ('imgid', imgid), ('ans', ans)],
													  format='tsv')
	batch_sizes = (1, 1)
	train_loader, val_loader = data.BucketIterator.splits((train_data, val_data), batch_sizes=batch_sizes, repeat=False, sort_key=lambda x: len(x.ques))
	ques.build_vocab(train_data)
	print('vocabulary size: {}'.format(len(ques.vocab.stoi)))
	return ques, train_loader, val_loader


def _dump_datasets(loader, outfile, sorted=False):
	examples = []
	for ex in loader:
		examples.append((
			ex.quesid.data[0],
			ex.ques[0].data.squeeze().cpu().numpy(),
			ex.ques[1][0],
			ex.imgid.data[0],
			ex.ans.data[0]))
	if not sorted:
		examples.sort(key=lambda ex: ex[2])
	with open(outfile, 'wb') as trainf:
		pickle.dump(examples, trainf)

def _dump_vocab(vocab, outfile):
	with open(outfile, 'w') as fout:
		for tok, idx in vocab:
			fout.write('{}\t{}'.format(tok, idx) + '\n')


def preprocess(data_dir, train_ques_file, train_ans_file, val_ques_file, val_ans_file):
	print('Preprocessing with data root dir: {}'.format(data_dir))

	train_tsv_file, val_tsv_file = 'train.tsv', 'val.tsv'
	print('Creating tsv datasets: {}, {}'.format(train_tsv_file, val_tsv_file))
	ansid = _create_tsv(data_dir=data_dir, quesfile=train_ques_file, ansfile=train_ans_file, outfile=train_tsv_file)


	_create_tsv(data_dir=data_dir, quesfile=val_ques_file, ansfile=val_ans_file, outfile=val_tsv_file, ansid=ansid)

	print('Creating loaders...')
	ques, train_loader, val_loader = _create_loaders(data_dir, train_tsv_file, val_tsv_file)

	ques_stoi_file = join(data_dir, 'ques_stoi.tsv')
	print('Dumping vocabulary to {}'.format(ques_stoi_file))
	_dump_vocab(ques.vocab.stoi.items(), ques_stoi_file)

	train_data_file, val_data_file = join(data_dir, 'train.pkl'), join(data_dir, 'val.pkl')
	print('Dumping train dataset to {}'.format(train_data_file))
	_dump_datasets(train_loader, outfile=train_data_file)
	print('Dumping val dataset to {}'.format(val_data_file))
	_dump_datasets(val_loader, outfile=val_data_file, sorted=True)

def preprocess_small(data_dir, train_ques_file, train_ans_file, val_ques_file, val_ans_file):
	
	print('Preprocessing the smaller dataset')

	train_tsv_file, val_tsv_file = 'train_small.tsv', 'val_small.tsv'
	print('Creating tsv datasets: {}, {}'.format(train_tsv_file, val_tsv_file))
	ansid = _create_small_tsv(data_dir=data_dir, quesfile=train_ques_file, ansfile=train_ans_file, outfile=train_tsv_file)

	_create_small_tsv(data_dir=data_dir, quesfile=val_ques_file, ansfile=val_ans_file, outfile=val_tsv_file, ansid=ansid)
	
	print('Creating loaders...')
	ques, train_loader, val_loader = _create_loaders(data_dir, train_tsv_file, val_tsv_file)

	############ Questions to Integers #############

	ques_stoi_file = join(data_dir, 'ques_stoi_small.tsv')
	print("Dumping vocabulary to {}".format(ques_stoi_file))
	_dump_vocab(ques.vocab.stoi.items(), ques_stoi_file)

	############ Training and Validation Dataset Files ##########################
	train_data_file, val_data_file = join(data_dir, 'train_small.pkl'), join(data_dir, 'val_small.pkl')
	print('Dumping train dataset to {}'.format(train_data_file))
	_dump_datasets(train_loader, outfile=train_data_file)
	print('Dumping val dataset to {}'.format(val_data_file))
	_dump_datasets(val_loader, outfile=val_data_file, sorted=True)



if __name__ == '__main__':

	data_dir = './data'

	train_ques_file = 'v2_OpenEnded_mscoco_val2014_questions.json'
	val_ques_file = 'v2_OpenEnded_mscoco_val2014_questions.json'

	train_ans_file = 'v2_mscoco_val2014_annotations.json'
	val_ans_file = 'v2_mscoco_val2014_annotations.json'

	# preprocess(data_dir, train_ques_file, train_ans_file, val_ques_file, val_ans_file)
	preprocess_small(data_dir, train_ques_file, train_ans_file, val_ques_file, val_ans_file)


# ansfile = 'v2_mscoco_val2014_annotations.json'
# ansfile = os.path.join(data_dir, ansfile)

# outfile = 'quesfile.json'

# ques_json = json.load(open(quesfile))
# ques = [q['question'] for q in ques_json['questions']]
# quesid = [q['question_id'] for q in ques_json['questions']]
# img_id = [q['image_id'] for q in ques_json['questions']]

# ans_json = json.load(open(ansfile))
# ans = [a['multiple_choice_answer'] for a in ans_json['annotations']]


# k = 1000
# c = Counter(ans)
# topk = c.most_common(n = k)
# ansid = dict((a[0], i) for i, a in enumerate(topk))
# ans_itos_file = os.path.join(data_dir, 'ans_itos.tsv')

# with open(ans_itos_file, 'w') as fout:
#     for i, (a, freq) in enumerate(topk):
#         fout.write('{}\t{}\t{}'.format(i, a, freq) + '\n')
#     fout.write('{}\t{}\t{}'.format(k, '<unk>', 'rest') + '\n')


# import pdb
# pdb.set_trace()

# ans_itos_file = os.path.join(data_dir, 'ans_itos.tsv')
# with open(ans_itos_file, 'w') as fout:
#     for i, (a, freq) in enumerate(topk):
#         fout.write('{}\t{}\t{}'.format(i, a, freq) + '\n')
#     fout.write('{}\t{}\t{}'.format(k, '<unk>', 'rest') + '\n')

# ans = [ansid[a] if a in ansid else k for a in ans]

# outfile = join(data_dir, outfile)
# with open(outfile, 'w') as out:
#     for q, qid, i, a in zip(ques, quesid, imgid, ans):
#         out.write('\t'.join([str(qid), q, str(i), str(a)]) + '\n')
