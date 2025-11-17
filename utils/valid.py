import tqdm
import torch
import utils.util as util
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_training_progress(train_losses, val_det_recalls, val_cls_accs, val_macro_precisions, 
                          val_macro_recalls, val_macro_f1s, epoch, save_dir, class_stats_history=None):
    """학습 진행 상황을 4x2 subplot으로 시각화하고 저장하는 함수 (Point-label 메트릭 전용 + 클래스별 성능)"""
    
    # 클래스별 통계 포함 여부에 따라 subplot 구성 변경
    if class_stats_history is not None and len(class_stats_history) > 0:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(15, 24))
    else:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # 1. Training Loss
    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Macro F1-score (주요 지표)
    ax2.plot(epochs_range, val_macro_f1s, 'darkgreen', linewidth=2, label='Macro F1-score ⭐')
    ax2.set_title('Macro F1-score (Primary Metric)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Detection Recall
    ax3.plot(epochs_range, val_det_recalls, 'cyan', linewidth=2, label='Detection Recall')
    ax3.set_title('Detection Recall (GT 중 찾은 비율)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Detection Recall')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Classification Accuracy
    ax4.plot(epochs_range, val_cls_accs, 'magenta', linewidth=2, label='Classification Accuracy')
    ax4.set_title('Classification Accuracy (분류 정확도)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Macro Precision & Recall
    ax5.plot(epochs_range, val_macro_precisions, 'orange', linewidth=2, label='Macro Precision')
    ax5.plot(epochs_range, val_macro_recalls, 'red', linewidth=2, label='Macro Recall')
    ax5.set_title('Macro Precision & Recall', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Score')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Detection & Classification 통합
    ax6.plot(epochs_range, val_det_recalls, 'cyan', linewidth=2, label='Detection Recall', alpha=0.7)
    ax6.plot(epochs_range, val_cls_accs, 'magenta', linewidth=2, label='Classification Accuracy', alpha=0.7)
    ax6.plot(epochs_range, val_macro_f1s, 'darkgreen', linewidth=3, label='Macro F1 ⭐')
    ax6.set_title('Overall Performance', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Score')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. 클래스별 F1-score (class_stats_history가 있을 경우)
    if class_stats_history is not None and len(class_stats_history) > 0:
        class_names = ['class_0', 'class_1+', 'class_2+', 'class_3+']
        class_colors = ['green', 'gold', 'blue', 'red']
        
        # 클래스별 F1 데이터 추출
        for class_idx, (class_name, color) in enumerate(zip(class_names, class_colors)):
            class_f1_values = []
            for stats in class_stats_history:
                if class_name in stats:
                    class_f1_values.append(stats[class_name]['f1'])
                else:
                    class_f1_values.append(0)
            ax7.plot(epochs_range, class_f1_values, color=color, linewidth=2, 
                    label=class_name, marker='o', markersize=3, alpha=0.8)
        
        ax7.set_title('Per-Class F1-Score', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('F1-Score')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        ax7.set_ylim([0, 1])
        
        # 8. 클래스별 Precision & Recall
        for class_idx, (class_name, color) in enumerate(zip(class_names, class_colors)):
            class_precision_values = []
            class_recall_values = []
            for stats in class_stats_history:
                if class_name in stats:
                    class_precision_values.append(stats[class_name]['precision'])
                    class_recall_values.append(stats[class_name]['recall'])
                else:
                    class_precision_values.append(0)
                    class_recall_values.append(0)
            
            # Precision (실선)
            ax8.plot(epochs_range, class_precision_values, color=color, linewidth=2, 
                    linestyle='-', label=f'{class_name} P', alpha=0.7)
            # Recall (점선)
            ax8.plot(epochs_range, class_recall_values, color=color, linewidth=2, 
                    linestyle='--', label=f'{class_name} R', alpha=0.7)
        
        ax8.set_title('Per-Class Precision (—) & Recall (- -)', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Score')
        ax8.grid(True, alpha=0.3)
        ax8.legend(ncol=2, fontsize=9)
        ax8.set_ylim([0, 1])
    
    # 전체 제목
    if class_stats_history is not None and len(class_stats_history) > 0:
        fig.suptitle(f'Training Progress (Point-Label Metrics + Per-Class Stats) - Epoch {epoch}', 
                     fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'Training Progress (Point-Label Metrics) - Epoch {epoch}', 
                     fontsize=16, fontweight='bold')
    
    # 최신 값들을 텍스트로 표시
    if len(train_losses) > 0:
        latest_info = f"""Latest Values (Epoch {len(train_losses)}):
Train Loss: {train_losses[-1]:.4f}
Macro F1: {val_macro_f1s[-1]:.4f} | Det Recall: {val_det_recalls[-1]:.4f} | Cls Acc: {val_cls_accs[-1]:.4f}
Macro Precision: {val_macro_precisions[-1]:.4f} | Macro Recall: {val_macro_recalls[-1]:.4f}
Best Macro F1: {max(val_macro_f1s):.4f} (Epoch {val_macro_f1s.index(max(val_macro_f1s))+1})"""
        
        # 클래스별 통계가 있으면 추가
        if class_stats_history is not None and len(class_stats_history) > 0:
            latest_class_stats = class_stats_history[-1]
            latest_info += "\n\nPer-Class F1 (Latest):"
            class_names = ['class_0', 'class_1+', 'class_2+', 'class_3+']
            for class_name in class_names:
                if class_name in latest_class_stats:
                    f1_val = latest_class_stats[class_name]['f1']
                    latest_info += f"\n  {class_name}: {f1_val:.4f}"
        
        fig.text(0.02, 0.02, latest_info, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.10)
    
    # 저장
    save_path = os.path.join(save_dir, f'training_progress_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📈 학습 진행 그래프 저장: {save_path}")
    
    plt.show()
    plt.close()


def visualize_ground_truth_and_prediction_separately(model, dataset, idx=0, conf_threshold=0.5, iou_threshold=0.3, epoch=None, save_dir=None):
    """실제 라벨과 예측 라벨을 subplot으로 좌우에 표시하는 함수"""
    if len(dataset) <= idx:
        print(f"경고: 데이터셋이 비어 있거나 idx {idx}가 데이터셋 크기({len(dataset)})보다 큽니다.")
        return
    
    model.eval()
    img, cls, box, _ = dataset[idx]
    
    # 하나의 figure에 2개의 subplot 생성 (1행 2열)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    img = img.cpu() / 255.
    # Subplot 1: Ground Truth (실제 라벨)
    ax1.imshow(img.permute(1, 2, 0).cpu().numpy())
    
    for i in range(len(cls)):
        class_id = cls[i].item()
        x_center, y_center, w, h = box[i].tolist()
        
        x = (x_center - w/2) * img.shape[2]
        y = (y_center - h/2) * img.shape[1]
        w_box = w * img.shape[2]
        h_box = h * img.shape[1]
        if class_id == 0: #class_0
            color = 'green'
        elif class_id == 1: #class_1+
            color = 'yellow'
        elif class_id == 2: #Class_2+
            color = 'blue'
        else: #Class_3+
            color = 'red'
        # 중심점 표시
        # 중심점 좌표 계산
        center_x = int(x + w_box / 2)
        center_y = int(y + h_box / 2)

        ax1.scatter(center_x, center_y, facecolors='none',  s=20, marker='o', edgecolors=color, linewidths=1)

    gt_title = f'Ground Truth '
    if epoch is not None:
        gt_title += f' - Epoch {epoch}'
    ax1.set_title(gt_title, fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Subplot 2: Model Prediction (예측 라벨)
    ax2.imshow(img.permute(1, 2, 0).cpu().numpy())
    
    prediction_count = 0
    with torch.no_grad():
        img_input = img.unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            pred = model(img_input)

        # NMS 적용
        results = util.non_max_suppression(pred, confidence_threshold=conf_threshold, iou_threshold=iou_threshold)
        if len(results[0]) > 0:
            for *xyxy, conf, cls_id in results[0]:
                x1, y1, x2, y2 = xyxy
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                w_pred = x2 - x1
                h_pred = y2 - y1
                
                if cls_id.item() == 0: #class_0
                    color = 'green'
                elif cls_id.item() == 1: #class_1+
                    color = 'yellow'
                elif cls_id.item() == 2: #Class_2+
                    color = 'blue'
                else: #Class_3+
                    color = 'red'
                
                center_x = (x1 + x2)//2
                center_y = (y1 + y2)//2
                ax2.scatter(center_x, center_y, facecolors='none',  s=20, marker='o', edgecolors=color, linewidths=1)

                prediction_count += 1
        
        if prediction_count == 0:
            ax2.text(img.shape[2]//2, img.shape[1]//2, 'No Predictions', 
                     fontsize=20, color='white', ha='center', va='center',
                     bbox=dict(facecolor='red', alpha=0.8, pad=10))
    
    pred_title = f'Model Prediction - {prediction_count} detections'
    if epoch is not None:
        pred_title += f' - Epoch {epoch}'
    ax2.set_title(pred_title, fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # 전체 figure 제목 설정
    if epoch is not None:
        fig.suptitle(f'Validation Comparison - Epoch {epoch}, Sample {idx+1}', 
                     fontsize=18, fontweight='bold', y=0.95)
    
    # 범례 추가
    legend_elements = [
        patches.Patch(color='green', label='Class_0'),
        patches.Patch(color='yellow', label='Class_1+'),
        patches.Patch(color='blue', label='Class_2+'),
        patches.Patch(color='red', label='Class_3+'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, 0.02), fontsize=12)
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
    # 저장
    if save_dir and epoch:
        save_path = os.path.join(save_dir, f'validation_comparison_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 비교 이미지 저장: {save_path}")
    
    # plt.show()
    plt.clf()
    
    
    
def compute_validation_metrics(model, val_loader, device, params):
    """검증 메트릭 계산 함수 (mAP, precision, recall 포함) - loss 계산 제거, 라벨 없는 경우 처리"""
    model.eval()
    
    # Configure IoU thresholds for mAP calculation
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).to(device)  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()
    
    metrics = []
    
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images = val_images.to(device).float() / 255
            _, _, h, w = val_images.shape  # batch-size, channels, height, width
            scale = torch.tensor((w, h, w, h)).to(device)
            
            # 모델 예측만 수행 (loss 계산 제거)
            with torch.amp.autocast('cuda'):
                val_outputs = model(val_images)
            
            # NMS for metric calculation
            # Point 라벨에 최적화된 threshold 사용
            outputs = util.non_max_suppression(val_outputs, confidence_threshold=0.25, iou_threshold=0.45)
            
            # Metrics calculation
            for i, output in enumerate(outputs):
                idx = val_targets['idx'] == i
                cls = val_targets['cls'][idx]
                box = val_targets['box'][idx]
                
                # 라벨도 없고 예측도 없는 경우 - 완전히 건너뛰기
                if cls.shape[0] == 0 and output.shape[0] == 0:
                    continue
                
                # 라벨은 없지만 예측이 있는 경우 (False Positives)
                if cls.shape[0] == 0 and output.shape[0] > 0:
                    metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)
                    metrics.append((metric, output[:, 4], output[:, 5], torch.tensor([], device=device)))
                    continue
                
                # 라벨은 있지만 예측이 없는 경우 (False Negatives)
                if cls.shape[0] > 0 and output.shape[0] == 0:
                    cls = cls.to(device)
                    metric = torch.zeros(0, n_iou, dtype=torch.bool).to(device)
                    metrics.append((metric, torch.zeros(0).to(device), torch.zeros(0).to(device), cls.squeeze(-1)))
                    continue
                
                # 라벨도 있고 예측도 있는 경우만 정상 처리
                cls = cls.to(device)
                box = box.to(device)
                
                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)
                
                # Evaluate - cls와 box가 모두 존재하는 경우만 처리
                try:
                    # cls 차원 확인 및 조정
                    if cls.dim() == 1:
                        cls_reshaped = cls.unsqueeze(1)  # [N] -> [N, 1]
                    else:
                        cls_reshaped = cls
                    
                    # box를 xyxy 형식으로 변환
                    box_xyxy = util.wh2xy(box) * scale
                    
                    # target 생성 [N, 5] (class, x1, y1, x2, y2)
                    target = torch.cat(tensors=(cls_reshaped, box_xyxy), dim=1)
                    metric = util.compute_metric(output[:, :6], target, iou_v)
                except Exception as e:
                    print(f"메트릭 계산 중 오류 (건너뛰기): {e}")
                    continue
                
                # Append
                metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))
    
    # Calculate mAP if we have metrics
    m_pre, m_rec, map50, mean_ap = 0, 0, 0, 0
    if len(metrics) > 0:
        try:
            # 각 메트릭 요소를 안전하게 결합
            stats = []
            for i in range(4):  # metric, conf, cls_pred, cls_true
                elements = []
                for metric_tuple in metrics:
                    if i < len(metric_tuple) and metric_tuple[i] is not None:
                        element = metric_tuple[i]
                        # 텐서를 numpy로 변환하고 차원 확인
                        if isinstance(element, torch.Tensor):
                            element_np = element.cpu().numpy()
                            # 0차원 텐서를 1차원으로 변환
                            if element_np.ndim == 0:
                                element_np = np.array([element_np])
                            elements.append(element_np)
                        else:
                            elements.append(element)
                
                # 요소들이 있을 때만 concatenate
                if elements:
                    # 모든 요소가 같은 차원인지 확인
                    if all(isinstance(elem, np.ndarray) for elem in elements):
                        try:
                            concatenated = np.concatenate(elements, axis=0)
                            stats.append(concatenated)
                        except ValueError as ve:
                            print(f"Concatenation 오류 (인덱스 {i}): {ve}")
                            stats.append(np.array([]))
                    else:
                        stats.append(np.array([]))
                else:
                    stats.append(np.array([]))
            
            # stats가 올바르게 생성되었는지 확인
            if len(stats) == 4 and all(isinstance(s, np.ndarray) for s in stats):
                tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*stats, plot=False, names=params["names"])
            else:
                print("메트릭 통계 생성 실패")
                m_pre, m_rec, map50, mean_ap = 0, 0, 0, 0
                
        except Exception as e:
            print(f"mAP 계산 중 오류: {e}")
            print(f"메트릭 개수: {len(metrics)}")
            if len(metrics) > 0:
                print(f"첫 번째 메트릭 구조: {[type(x) for x in metrics[0]]}")
                print(f"첫 번째 메트릭 크기: {[x.shape if hasattr(x, 'shape') else len(x) if hasattr(x, '__len__') else 'scalar' for x in metrics[0]]}")
            m_pre, m_rec, map50, mean_ap = 0, 0, 0, 0
    
    return m_pre, m_rec, map50, mean_ap


def compute_validation_metrics_with_kappa(model, val_loader, device, params):
    """Cohen's Kappa를 포함한 검증 메트릭 계산 - 개선 버전 (배경 없음, 4개 세포 클래스만)"""
    try:
        from sklearn.metrics import cohen_kappa_score
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("경고: scikit-learn 또는 scipy가 설치되지 않아 Cohen's Kappa를 계산할 수 없습니다.")
        precision, recall, map50, mean_ap = compute_validation_metrics(model, val_loader, device, params)
        return precision, recall, map50, mean_ap, 0.0
    
    # 기본 메트릭 계산
    precision, recall, map50, mean_ap = compute_validation_metrics(model, val_loader, device, params)
    
    # Cohen's Kappa 계산 - 객체 매칭 기반
    # 배경이 없으므로 매칭된 객체만 사용
    model.eval()
    all_gt_classes = []
    all_pred_classes = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device).float() / 255
            
            # 예측
            with torch.amp.autocast('cuda'):
                pred = model(images)
            
            # NMS 적용
            results = util.non_max_suppression(pred, confidence_threshold=0.25, iou_threshold=0.45)
            
            # 각 이미지에 대해 처리
            for i in range(len(images)):
                # Ground truth
                cls_targets = targets['cls']
                box_targets = targets['box']
                idx_targets = targets['idx']
                
                batch_mask = idx_targets == i
                if not batch_mask.any():
                    continue
                
                batch_cls = cls_targets[batch_mask].cpu().numpy()
                batch_box = box_targets[batch_mask].cpu().numpy()
                
                # Predictions
                if len(results) > i and len(results[i]) > 0:
                    pred_boxes = results[i][:, :4].cpu().numpy()  # xyxy
                    pred_classes = results[i][:, 5].cpu().numpy()  # class
                    
                    # GT box를 xyxy로 변환
                    gt_boxes_xyxy = []
                    for box in batch_box:
                        x_center, y_center, w, h = box
                        x1 = (x_center - w/2) * 512
                        y1 = (y_center - h/2) * 512
                        x2 = (x_center + w/2) * 512
                        y2 = (y_center + h/2) * 512
                        gt_boxes_xyxy.append([x1, y1, x2, y2])
                    gt_boxes_xyxy = np.array(gt_boxes_xyxy)
                    
                    # IoU 행렬 계산
                    iou_matrix = compute_iou_matrix(gt_boxes_xyxy, pred_boxes)
                    
                    # Hungarian Algorithm으로 최적 매칭
                    if iou_matrix.size > 0 and iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
                        gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
                        
                        # IoU 임계값 이상인 매칭만 사용
                        iou_threshold = 0.3
                        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                            if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
                                all_gt_classes.append(int(batch_cls[gt_idx]))
                                all_pred_classes.append(int(pred_classes[pred_idx]))
                        
                        # 💡 배경이 없으므로 매칭되지 않은 GT와 Pred는 Kappa 계산에서 제외
                        # False Negative/Positive는 Precision/Recall에서 처리됨
    
    # Cohen's Kappa 계산 (매칭된 객체만 사용)
    try:
        if len(all_gt_classes) > 0 and len(all_pred_classes) > 0:
            kappa = cohen_kappa_score(all_gt_classes, all_pred_classes)
        else:
            kappa = 0.0
            print("경고: 매칭된 객체가 없어 Kappa를 계산할 수 없습니다.")
    except Exception as e:
        print(f"Cohen's Kappa 계산 오류: {e}")
        kappa = 0.0
    
    return precision, recall, map50, mean_ap, kappa


def compute_iou_matrix(boxes1, boxes2):
    """
    두 박스 집합 간의 IoU 행렬 계산
    boxes: [N, 4] (x1, y1, x2, y2)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    # 면적 계산
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # IoU 행렬
    iou_matrix = np.zeros((len(boxes1), len(boxes2)))
    
    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            # 교집합
            x1 = max(boxes1[i, 0], boxes2[j, 0])
            y1 = max(boxes1[i, 1], boxes2[j, 1])
            x2 = min(boxes1[i, 2], boxes2[j, 2])
            y2 = min(boxes1[i, 3], boxes2[j, 3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                union = area1[i] + area2[j] - intersection
                iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    return iou_matrix


def get_kappa_interpretation(kappa):
    """Kappa 값 해석"""
    if kappa < 0: 
        return "Poor"
    elif kappa < 0.21: 
        return "Slight"
    elif kappa < 0.41: 
        return "Fair"  
    elif kappa < 0.61: 
        return "Moderate"
    elif kappa < 0.81: 
        return "Substantial"
    else: 
        return "Almost Perfect"


def quick_kappa_test(model, val_loader, device):
    """현재 모델의 Cohen's Kappa 빠른 측정"""
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("경고: scikit-learn이 설치되지 않아 Cohen's Kappa를 계산할 수 없습니다.")
        return 0.0
        
    model.eval()
    
    # 몇 개 샘플로 빠른 테스트
    sample_gt = []
    sample_pred = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i >= 10:  # 10개 배치만 테스트
                break
                
            images = images.to(device).float() / 255
            pred = model(images)
            results = util.non_max_suppression(pred, confidence_threshold=0.25, iou_threshold=0.45)
            
            # 간단한 비교를 위해 객체 개수 기반 라벨링
            gt_count = len(targets['cls'])
            pred_count = len(results[0]) if len(results) > 0 and len(results[0]) > 0 else 0
            
            # 단순화된 라벨 (0: 없음, 1: 적음, 2: 많음)
            gt_label = 0 if gt_count == 0 else (1 if gt_count <= 5 else 2)
            pred_label = 0 if pred_count == 0 else (1 if pred_count <= 5 else 2)
            
            sample_gt.append(gt_label)
            sample_pred.append(pred_label)
    
    try:
        if len(sample_gt) > 0 and len(sample_pred) > 0:
            quick_kappa = cohen_kappa_score(sample_gt, sample_pred)
        else:
            quick_kappa = 0.0
    except Exception as e:
        print(f"빠른 Kappa 계산 오류: {e}")
        quick_kappa = 0.0
    
    print(f"📊 빠른 Cohen's Kappa 측정: {quick_kappa:.4f} ({get_kappa_interpretation(quick_kappa)})")
    return quick_kappa


def compute_distance_matrix(centers1, centers2):
    """
    두 중심점 집합 간의 Euclidean 거리 행렬 계산
    centers: [N, 2] (x, y)
    """
    if len(centers1) == 0 or len(centers2) == 0:
        return np.zeros((len(centers1), len(centers2)))
    
    # 거리 행렬 계산 (브로드캐스팅 사용)
    centers1 = np.array(centers1).reshape(-1, 2)
    centers2 = np.array(centers2).reshape(-1, 2)
    
    # Euclidean 거리: sqrt((x1-x2)^2 + (y1-y2)^2)
    diff = centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :]  # [N1, N2, 2]
    distances = np.sqrt(np.sum(diff**2, axis=2))  # [N1, N2]
    
    return distances


def compute_point_label_metrics(model, val_loader, device, params, distance_threshold=16):
    """
    Point-label에 최적화된 검증 메트릭 계산
    - Distance-based matching (IoU 대신 중심점 거리 사용)
    - Detection recall: GT 세포를 얼마나 찾았는가
    - Classification accuracy: 찾은 세포의 클래스를 얼마나 정확하게 분류했는가
    
    Args:
        model: YOLO 모델
        val_loader: 검증 데이터로더
        device: 디바이스
        params: 파라미터 (클래스 이름 등)
        distance_threshold: 매칭 거리 임계값 (픽셀 단위, 기본 16px)
    
    Returns:
        dict: {
            'detection_recall': GT 중 매칭된 비율,
            'classification_accuracy': 매칭된 객체 중 올바르게 분류된 비율,
            'macro_precision': 클래스별 평균 정밀도,
            'macro_recall': 클래스별 평균 재현율,
            'macro_f1': 클래스별 평균 F1,
            'overall_recall': 전체 재현율,
            'class_stats': 클래스별 상세 통계
        }
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("경고: scipy가 설치되지 않아 Point-label 메트릭을 계산할 수 없습니다.")
        return {}
    
    model.eval()
    
    # 전체 통계
    total_gt = 0
    total_matched = 0
    total_correct_class = 0
    
    # 클래스별 통계 (4개 클래스: 0, 1+, 2+, 3+)
    num_classes = 4
    class_tp = np.zeros(num_classes)  # True Positive (올바르게 탐지+분류)
    class_fp = np.zeros(num_classes)  # False Positive (잘못 탐지 또는 잘못 분류)
    class_fn = np.zeros(num_classes)  # False Negative (탐지 실패)
    class_gt_count = np.zeros(num_classes)  # GT 개수
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device).float() / 255
            
            # 예측
            with torch.amp.autocast('cuda'):
                pred = model(images)
            
            # NMS 적용
            results = util.non_max_suppression(pred, confidence_threshold=0.25, iou_threshold=0.45)
            
            # 각 이미지에 대해 처리
            for i in range(len(images)):
                # Ground truth 추출
                cls_targets = targets['cls']
                box_targets = targets['box']
                idx_targets = targets['idx']
                
                batch_mask = idx_targets == i
                if not batch_mask.any():
                    continue
                
                batch_cls = cls_targets[batch_mask].cpu().numpy()
                batch_box = box_targets[batch_mask].cpu().numpy()
                
                # GT 중심점 계산 (normalized -> pixel)
                img_size = 512  # 이미지 크기
                gt_centers = []
                for box in batch_box:
                    x_center = box[0] * img_size
                    y_center = box[1] * img_size
                    gt_centers.append([x_center, y_center])
                gt_centers = np.array(gt_centers)
                
                # GT 클래스별 카운트
                for cls_id in batch_cls:
                    class_gt_count[int(cls_id)] += 1
                
                total_gt += len(batch_cls)
                
                # Predictions 처리
                if len(results) > i and len(results[i]) > 0:
                    pred_boxes = results[i][:, :4].cpu().numpy()  # xyxy
                    pred_classes = results[i][:, 5].cpu().numpy()  # class
                    
                    # Prediction 중심점 계산
                    pred_centers = []
                    for box in pred_boxes:
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        pred_centers.append([x_center, y_center])
                    pred_centers = np.array(pred_centers)
                    
                    # 거리 행렬 계산
                    distance_matrix = compute_distance_matrix(gt_centers, pred_centers)
                    
                    # Hungarian Algorithm으로 최적 매칭
                    if distance_matrix.size > 0 and distance_matrix.shape[0] > 0 and distance_matrix.shape[1] > 0:
                        gt_indices, pred_indices = linear_sum_assignment(distance_matrix)
                        
                        # 거리 임계값 이하인 매칭만 사용
                        matched_gt = set()
                        matched_pred = set()
                        
                        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                            if distance_matrix[gt_idx, pred_idx] <= distance_threshold:
                                matched_gt.add(gt_idx)
                                matched_pred.add(pred_idx)
                                total_matched += 1
                                
                                gt_cls = int(batch_cls[gt_idx])
                                pred_cls = int(pred_classes[pred_idx])
                                
                                # 클래스가 일치하면 TP
                                if gt_cls == pred_cls:
                                    total_correct_class += 1
                                    class_tp[gt_cls] += 1
                                else:
                                    # 클래스 불일치: GT는 FN, Pred는 FP
                                    class_fn[gt_cls] += 1
                                    class_fp[pred_cls] += 1
                        
                        # 매칭되지 않은 GT: False Negative
                        for gt_idx in range(len(batch_cls)):
                            if gt_idx not in matched_gt:
                                gt_cls = int(batch_cls[gt_idx])
                                class_fn[gt_cls] += 1
                        
                        # 매칭되지 않은 Pred: False Positive
                        for pred_idx in range(len(pred_classes)):
                            if pred_idx not in matched_pred:
                                pred_cls = int(pred_classes[pred_idx])
                                class_fp[pred_cls] += 1
                    else:
                        # 매칭 불가능: 모든 GT는 FN
                        for cls_id in batch_cls:
                            class_fn[int(cls_id)] += 1
                else:
                    # 예측 없음: 모든 GT는 FN
                    for cls_id in batch_cls:
                        class_fn[int(cls_id)] += 1
    
    # 메트릭 계산
    detection_recall = total_matched / total_gt if total_gt > 0 else 0
    classification_accuracy = total_correct_class / total_matched if total_matched > 0 else 0
    
    # 클래스별 메트릭
    class_precision = []
    class_recall = []
    class_f1 = []
    
    for c in range(num_classes):
        # Precision = TP / (TP + FP)
        precision = class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) > 0 else 0
        
        # Recall = TP / (TP + FN)
        recall = class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) > 0 else 0
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_precision.append(precision)
        class_recall.append(recall)
        class_f1.append(f1)
    
    # Macro-averaged 메트릭 (클래스별 평균)
    macro_precision = np.mean(class_precision)
    macro_recall = np.mean(class_recall)
    macro_f1 = np.mean(class_f1)
    
    # Overall Recall (전체 재현율)
    overall_recall = np.sum(class_tp) / np.sum(class_tp + class_fn) if np.sum(class_tp + class_fn) > 0 else 0
    
    # 클래스별 상세 통계
    class_names = params.get('names', {})
    class_stats = {}
    for c in range(num_classes):
        class_name = class_names.get(c, f'Class_{c}')
        class_stats[class_name] = {
            'precision': class_precision[c],
            'recall': class_recall[c],
            'f1': class_f1[c],
            'tp': int(class_tp[c]),
            'fp': int(class_fp[c]),
            'fn': int(class_fn[c]),
            'gt_count': int(class_gt_count[c])
        }
    
    return {
        'detection_recall': detection_recall,
        'classification_accuracy': classification_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'overall_recall': overall_recall,
        'class_stats': class_stats
    }