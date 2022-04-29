import torch
import torch.nn.functional as F
from model.transformer import Transformer
from dataProcess.processing import abs_test_iter, def_test_iter
from dataProcess.processing import abs_train_iter, def_train_iter


def train_model(abs_train_data, abs_test_data, def_train_data, def_test_data, model_info):
    model_info = model_info.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.6)
    model_info.train()
    best_acc = 0
    print('training...')
    for epoch in range(1, 6):
        model_info.train()
        total_loss = 0.0
        accuracy = 0
        n = 0
        for i, data in enumerate(zip(abs_train_data, def_train_data)):
            abs_x = data[0][0].to(device)
            def_x = data[1][0].to(device)
            y_label = data[1][1].to(device)
            # 梯度初始化0
            optimizer.zero_grad()
            y_hat = model_info(abs_x, def_x)
            loss = F.binary_cross_entropy(y_hat, y_label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            accuracy += float(torch.sum(torch.argmax(y_hat, dim=1) == y_label))
            n += y_label.shape[0]
        print('>>> Epoch_{}, Train loss is {}, Accuracy:{} \n'.format(epoch, total_loss / n,
                                                                      accuracy / n))
        model_info.eval()
        total_loss = 0.0
        accuracy = 0
        n = 0
        for i, data in enumerate(zip(abs_test_data, def_test_data)):
            if isinstance(model_info, torch.nn.Module):
                # 启动评估模式
                model_info.eval()
                # 测试数据的标签
                test_label = data[0][1]
                # 根据当前训练轮数的训练参数来使用测试集对模型效果进行拟合的结果
                test_hat = model_info(data[0][0].to(device), data[1][0].to(device))
                # 准确率求和
                accuracy += (test_hat.argmax(dim=1) == test_label).sum().cpu().item()
                model_info.train()
            n += data[0][1].shape[0]
        print('>>> Epoch_{}, Accuracy:{} \n'.format(epoch, accuracy / n))
        if accuracy / n > best_acc:
            print('save model...')
            best_acc = accuracy / n
            torch.save(model, '../modelData/transformer.pkl')


name = 'Transformer'
model = Transformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    train_model(abs_train_iter, abs_test_iter, def_train_iter, def_test_iter, model)
