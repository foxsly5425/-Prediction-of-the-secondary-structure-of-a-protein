output_dim = 3

model     = ProteinLSTM(embedding_layer, output_dim)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

wandb.init(project="SecondaryStructures2", config={
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": optimizer.param_groups[0]["lr"],
    "num_classes": output_dim
})

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

scaler = torch.cuda.amp.GradScaler()

model_epochs = 20

for epoch in range(model_epochs):
    start_epoch = time.time()

    model.train()
    train_total_loss  = 0
    correct_train     = 0
    total_train       = 0
    y_true_train      = []
    y_pred_train      = []
    train_loader_tqdm = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{model_epochs} - Обучение", unit="batch")

    for embeddings, labels in train_loader_tqdm:
        embeddings, labels = embeddings.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(embeddings)  
            batch_size_curr, seq_len, num_classes = outputs.shape

            outputs_flat = outputs.view(-1, num_classes) 
            labels_flat  = labels.view(-1)                
            mask         = labels_flat != -1             
            loss         = criterion(outputs_flat[mask], labels_flat[mask])
            _, predict   = torch.max(outputs, 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_total_loss += loss.item()
        correct_train    += (predict.view(-1)[mask] == labels_flat[mask]).sum().item()
        total_train      += mask.sum().item()
        y_true_batch      = labels_flat[mask].cpu().numpy()
        y_pred_batch      = predict.view(-1)[mask].cpu().numpy()
        y_true_train.extend(y_true_batch)
        y_pred_train.extend(y_pred_batch)
        train_loader_tqdm.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Accuracy': f"{(correct_train / total_train * 100):.2f}%"
        })

    precision_train   = precision_score(y_true_train, y_pred_train, average='macro')
    recall_train      = recall_score(y_true_train, y_pred_train, average='macro')
    f1_train          = f1_score(y_true_train, y_pred_train, average='macro')
    train_total_loss /= len(train_loader)
    accuracy_train    = correct_train / total_train * 100

    model.eval()
    total_test_loss = 0
    correct_test    = 0
    total_test      = 0
    y_true_test     = []
    y_pred_test     = []

    test_loader_tqdm = tqdm(test_loader, desc=f"Эпоха {epoch+1}/{model_epochs} - Тестирование", unit="batch")

    with torch.no_grad():
        for embeddings, labels in test_loader_tqdm:
            embeddings, labels = embeddings.to(device), labels.to(device)

            outputs = model(embeddings)
            batch_size_curr, seq_len, num_classes = outputs.shape

            outputs_flat = outputs.view(-1, num_classes)
            labels_flat  = labels.view(-1)
            mask         = labels_flat != -1
            loss         = criterion(outputs_flat[mask], labels_flat[mask])
            _, predict   = torch.max(outputs, 2)

            total_test_loss += loss.item()
            correct_test    += (predict.view(-1)[mask] == labels_flat[mask]).sum().item()
            total_test      += mask.sum().item()
            y_true_batch     = labels_flat[mask].cpu().numpy()
            y_pred_batch     = predict.view(-1)[mask].cpu().numpy()
            y_true_test.extend(y_true_batch)
            y_pred_test.extend(y_pred_batch)

            # Обновление информации в прогресс-баре
            test_loader_tqdm.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Accuracy': f"{(correct_test / total_test * 100):.2f}%"
            })

    precision_test   = precision_score(y_true_test, y_pred_test, average='macro')
    recall_test      = recall_score(y_true_test, y_pred_test, average='macro')
    f1_test          = f1_score(y_true_test, y_pred_test, average='macro')
    total_test_loss /= len(test_loader)
    accuracy_test    = correct_test / total_test * 100

    scheduler.step(total_test_loss)

    end_epoch  = time.time()
    time_epoch = round(end_epoch - start_epoch)

    print(f'Epoch - {epoch+1}/{model_epochs}, Training Loss - {train_total_loss:.4f}, Testing Loss - {total_test_loss:.4f}, Time - {time_epoch} sec')
    print(f'                  Accuracy on Train - {accuracy_train:.2f}%, Accuracy on Test - {accuracy_test:.2f}%')
    print(f'                  F1 on Train - {f1_train:.4f}, F1 on Test - {f1_test:.4f}')
    print(f'                  Precision on Train - {precision_train:.4f}, Precision on Test - {precision_test:.4f}')
    print(f'                  Recall on Train - {recall_train:.4f}, Recall on Test - {recall_test:.4f}')
    # print(f'                  Learning Rate - {current_lr:.6f}')



    wandb.log({
        "train_loss": train_total_loss,
        "train_accuracy": accuracy_train,
        # "train_f1": f1_train,
        # "train_precision": precision_train,
        # "train_recall": recall_train,
        "val_loss": total_test_loss,
        "val_accuracy": accuracy_test,
        # "val_f1": f1_test,
        # "val_precision": precision_test,
        # "val_recall": recall_test,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch": epoch + 1,
    })


torch.save(model.state_dict(), 'protein_resnet_model.pth')
print("Модель сохранена в 'protein_resnet_model.pth'.")
