import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score
)

class MetricUtils:
    @staticmethod
    def compute_ACC(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        Input:
            pred_logit
            label
        Return:
            acc
        """
        pred=(torch.sigmoid(pred_logit)>=0.5).float()
        acc=(pred==label.float()).float().mean().item()
        return acc

    @staticmethod
    def compute_pos_recall(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        )->float:
        """
        실제 양성 중 양성으로 분류된 비율. 실제 양성이 없으면 NaN.
        """
        pred=torch.sigmoid(pred_logit.detach().reshape(-1))>=0.5
        positive=label.reshape(-1)==1
        return pred[positive].float().mean().item()

    @staticmethod
    def compute_neg_recall(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        )->float:
        """
        실제 음성 중 음성으로 분류된 비율. 실제 음성이 없으면 NaN.
        """
        pred=torch.sigmoid(pred_logit.detach().reshape(-1))>=0.5
        negative=label.reshape(-1)==0
        return (~pred[negative]).float().mean().item()

    @staticmethod
    def compute_balanced_ACC(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        )->float:
        """
        양성과 음성 재현율의 평균. 두 클래스가 모두 있어야 계산한다.
        """
        pred=(torch.sigmoid(pred_logit.detach())>=0.5).cpu().reshape(-1).numpy()
        true_label=label.detach().cpu().reshape(-1).numpy()
        if not ((true_label==0).any() and (true_label==1).any()):
            return float("nan")
        return float(balanced_accuracy_score(true_label,pred))

    @staticmethod
    def compute_AP(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        AP (Average Precision)
        -> 점수 상위에 실제 연결이 얼마나 잘 모여 있는지 평가.
        값 범위: 0 ~ 1
            1.0: 실제 연결들이 모든 가짜 연결보다 높은 점수를 받음.
            높을수록: 상위 예측에서 실제 연결을 더 정확하게 찾아냄.
            무작위 예측의 기준 수준: 평가 데이터의 양성 비율. 현재처럼 양성·음성이 1:1이면 약 0.5.
        """
        pred_score=torch.sigmoid(pred_logit.detach()).cpu().reshape(-1).numpy()
        true_label=label.detach().cpu().reshape(-1).numpy()
        return float(average_precision_score(true_label,pred_score))

    @staticmethod
    def compute_f1(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        )->float:
        """
        양성과 음성 F1의 평균. 두 클래스가 모두 있어야 계산한다.
        """
        pred=(torch.sigmoid(pred_logit.detach())>=0.5).cpu().reshape(-1).numpy()
        true_label=label.detach().cpu().reshape(-1).numpy()
        if not ((true_label==0).any() and (true_label==1).any()):
            return float("nan")
        return float(f1_score(true_label,pred,labels=[0,1],average="macro",zero_division=0))
