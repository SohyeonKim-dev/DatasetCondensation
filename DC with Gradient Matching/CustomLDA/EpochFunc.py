# 중요한 function! 
# epoch을 돌면서, loss와 정확도의 평균을 반환
# criterion = nn.CrossEntropyLoss().to(args.device)

def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    # dataloader의 문제..! epoch이 호출되는 지점을 다시 보아야한다! 
    # print(dataloader)
    # 한 번 씩 밖에 안 돈다 .. ! 
    # lda_X = []
    # lda_Y = []

    # print("epoch loop의 시작")
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    # 이 epoch에서 criterion 정의할 때
    # 즉 마지막에 cross entropy loss 정의하는 부분에서 LDA loss가 적용되어야 . . ? 
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
        # model을 학습 Mode로 설정한 것 
    else:
        net.eval()

    # dataset augmentation
    for din, datum in enumerate(dataloader):
        # print(din)
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        
        # lab -> label, y값을 의미함 
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        # lda_Y.append(copy.deepcopy(lab.detach()))

        # convNet에 image를 통과시킨 output
        
        # print(img.shape)
        # print(img)
        
        output = net(img) 
        # lda_X.append(copy.deepcopy(output.detach()))

        # 여기가 cross entropy loss
        # print(output.shape)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        # 각각의 loss를 합산 
        loss_avg += loss.item() * n_b 
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # print(output.detach().shape) # [10, 10]
    # print(lab.detach().shape) # [10]
    # print(len(lda_Y))

    # 여기서 batch image에 대한 평균을 모아서 출력하네
    # 여기에 람다 적용된 LDA Loss를 더하여 구현하자
    # LDA_Loss = (between class variance) / (within class variance)

    # if (din >= 1):
    #     print(len(lda_X)) # 40
    #     print(len(lda_Y)) # 40

    # print("epoch loop의 끝")

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg
