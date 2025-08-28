# check_align.py
import cv2, Levenshtein, torch
from model import CRNN
charset = ['-','O','P','1','2','/','4','0','3','A','D','C','@','E']

model = CRNN(imgH=32, nc=1, nclass=len(charset), nh=256).eval()
model.load_state_dict(torch.load('/root/workspace/checkpoints/crnn_best_finetuned.pth', map_location='cpu'))

for line in open('train.txt'):
    img_path, label = line.rstrip('\n').split('\t', 1)
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (100, 32))
    img = torch.from_numpy(img.astype('float32') / 255.).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(img).softmax(-1).squeeze().numpy()
    pred = ''.join([charset[c] for c in logits.argmax(1)])
    pred = ''.join([c for c in pred if c != '-'])
    if Levenshtein.distance(pred, label.strip()) > 0:
        print(img_path, 'pred:', pred, 'label:', label.strip())