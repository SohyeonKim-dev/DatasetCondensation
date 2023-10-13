def LDALoss(X, Y):
    # 네트워크 후반부에 배치되었다고 가정 // 이렇게 복잡한 함수가 잘 동작할까?
    # 서로 같은 Class 간 Var / 서로 다른 class (전체 data의 var)간 loss
    # X = last feature map (network를 통과한 image라고 생각해보자) / Y = label 
    # var = (X - m)^2 / n
    
    num_samples, num_features = X.size()    
    num_classes = len(torch.unique(Y))

    overall_mean = torch.mean(X, dim=0)
    allClassVar = torch.sum((X - overall_mean) ** 2, dim=0)

    # Compute class per variance (withinClassVar)

    withinClassVar = torch.zeros(num_features, device=X.device)
    for c in range(num_classes):
        class_samples = X[Y == c]
        class_mean = torch.mean(class_samples, dim=0)
        diff = class_samples - class_mean
        withinClassVar += torch.sum(diff * diff, dim=0)

    LDALoss = 1 - ( allClassVar / withinClassVar )

    return LDALoss
