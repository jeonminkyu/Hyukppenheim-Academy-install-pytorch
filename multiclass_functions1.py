import torch
import numpy as np # confusion matrix 사용시
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
DEVICE =("cuda" if torch.cuda.is_available() else "cpu")
#로컬에서 mac gpu or cuda or cpu사용 코드
# DEVICE = (
#     "mps"
#     if torch.backends.mps.is_available()
#     else ("cuda" if torch.cuda.is_available() else "cpu")
# )

#수정 코드(학습시 진전도 확인 추가)
# from tqdm import tqdm
# import time

# def Train(model, train_DL, criterion, optimizer, EPOCH):
#     NoT = len(train_DL.dataset)
#     loss_history = []

#     model.train()
#     for ep in range(EPOCH):
#         rloss = 0
#         start = time.time()

#         pbar = tqdm(total=len(train_DL), desc=f"Epoch {ep+1}/{EPOCH}", ncols=100)
#         for step, (x_batch, y_batch) in enumerate(train_DL, 1):
#             x_batch = x_batch.to(DEVICE)
#             y_batch = y_batch.to(DEVICE)

#             y_hat = model(x_batch)
#             loss = criterion(y_hat, y_batch)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             loss_b = loss.item() * x_batch.size(0)
#             rloss += loss_b

#             avg_loss = rloss / NoT
#             elapsed = time.time() - start
#             ms_per_step = (elapsed / step) * 1000

#             pbar.set_postfix(loss=f"{avg_loss:.4e}", ms=f"{ms_per_step:.1f}")
#             pbar.update(1)

#         pbar.close()
#         loss_history.append(rloss / NoT)

#     return loss_history




#원본코드
def Train(model, train_DL, criterion, optimizer, EPOCH):
   
    NoT=len(train_DL.dataset) # Number of training data
    loss_history = []

    model.train() # train mode로! #torch에는 tensorflow처럼 .fit()같은 단일 학습 함수가 없다.
    for ep in range(EPOCH):
        rloss = 0
        pbar = tqdm(train_DL, desc=f"Epoch {ep+1}/{EPOCH}", leave=False)
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # cross entropy loss
            loss = criterion(y_hat, y_batch)
            # update
            optimizer.zero_grad() # gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # weight update
            # loss accumulation
            loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE 로 하면 마지막 18개도 32개로 계산해버림
            rloss += loss_b # running loss
            pbar.set_postfix(loss=rloss / NoT)
        # print loss
        loss_e = rloss/NoT # epoch loss
        loss_history += [loss_e]
        print(f"Epoch: {ep+1}, train loss: {loss_e:.3f}")
        print("-"*20)
        
    return loss_history

def Test(model,test_DL):
    model.eval()
    with torch.no_grad():
        rcorrect = 0
        # rloss = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # corrects accumulation
            pred = y_hat.argmax(dim=1)
            corrects_b = torch.sum(pred == y_batch).item()
            rcorrect += corrects_b
        accuracy_e = rcorrect/len(test_DL.dataset)*100
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({accuracy_e:.1f} %)")
    return round(accuracy_e,1)

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r") 

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num


def get_conf(model, test_DL):
    N = len(test_DL.dataset.classes)
    model.eval()
    with torch.no_grad():
        confusion = torch.zeros(N,N)
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # accuracy
            pred = y_hat.argmax(dim=1)
            
            confusion += torch.bincount(N * y_batch.cpu() + pred.cpu(), minlength=N**2).reshape(N, N)
        
    confusion = confusion.numpy()

    return confusion

def plot_confusion_matrix(confusion, classes=None):
    N = confusion.shape[0]
    accuracy=np.trace(confusion)/np.sum(confusion) * 100
    
    plt.figure(figsize=(10,7))
    plt.imshow(confusion, cmap="Blues")
    plt.title("confusion matrix")
    plt.colorbar()

    for i in range(N):
        for j in range(N):
            plt.text(j,i, round(confusion[i,j]), 
                     horizontalalignment="center", fontsize=10,
                     color="white" if confusion[i,j] > np.max(confusion) / 1.5 else "black")

    if classes is not None:
        plt.xticks(range(N), classes)
        plt.yticks(range(N), classes)
    else:
        plt.xticks(range(N))
        plt.yticks(range(N))

    plt.xlabel(f"Predicted label \n accuracy = {accuracy:.1f} %")
    plt.ylabel("True label")



