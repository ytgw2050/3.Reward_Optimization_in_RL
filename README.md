# 3.Reward_Optimization_in_RL
### Our study introduces a deep learning-based reward prediction model that efficiently combines the strengths of Sparse and Dense rewards, outperforming both in a stock trading simulation with consistent and superior stability.

강화학습의 핵심 구성요소 중 하나인 보상 함수의 설계는 에이전트의 학습 성능과 일반화 능력에 큰 영향을 미친다.  
지금까지 이루어진 연구에서는 Sparse 와 Dense 보상 함수의 장단점을 통합하여 에이전트의 성능을 향상시키려는 다양한 방법론이 제시되었지만, 
그러한 결합 방식들은 대체로 보상 설계자의 주관적 판단에 크게 의존하는 경향이 있어 결과적으로 에이전트의 일반화 성능을 제한하게 된다.

따라서 본 연구에서는 이러한 문제를 해결하기 위해 딥러닝 기반의 새로운 보상 예측 모델을 제안하고자 한다. 
연구에서 제시된 보상 예측 모델은 Sparse 보상 함수로 사전 훈련된 여러 모델들의 상태와 행동을 입력으로 받아, 각 스텝에서 경쟁 모델들의 최종 보상을 예측한다. 
이러한 보상 예측 모델을 하나의 밀도높은 보상 함수로 사용함으로써 Dense 보상 함수의 빠른 수렴성과 Sparse 보상 함수의 장기적 목표 지향성을 결합한 효율적인 학습 전략을 구축하였다.
