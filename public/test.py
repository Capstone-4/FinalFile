import sys
import json
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

# JSON 데이터를 받아와서 계산 수행
def perform_calculation(json_data):

    df = json_data
    
    df["total_floor"] = None
    df["current_floor"] = None

    for i in range(len(df)):
        flr_info = df.loc[i, "flrInfo"]
        tmp = flr_info.split("/")
        df.loc[i, "current_floor"] = tmp[0]
        df.loc[i, "total_floor"] = tmp[1]
    df.drop(["flrInfo"], axis = 1, inplace = True)

    df['isBase'] = df['current_floor'].apply(lambda x: 1 if str(x).startswith('B') else 0)
    df['current_floor'] = df['current_floor'].apply(lambda x: float(x[1:]) if str(x).startswith('B') else float(x))

    import torch
    from torch import nn

    class MLP(torch.nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4, 121),
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(121, 121),
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(121, 121),
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(121, 14),
            )
        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits  
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP().to(device)
    model_dic = torch.load("MLP_model.pth", map_location=device)

    model.load_state_dict(model_dic)
    
    input = df[["isBase", "current_floor", "spc", "total_floor"]].astype(float).to_numpy()
    input = torch.tensor(input)


    model.eval()
    with torch.no_grad():
        input = input.to(device, dtype=torch.float)
        pred = model(input)
        probabilities = F.softmax(pred, dim=1)

    probabilities_percent = probabilities * 100
    torch.set_printoptions(precision=2, sci_mode=False)

    # Tensor: probabilities_percent[0]
    import numpy as np

    def get_top_3_values(arr):
        # NumPy 배열로 변환
        arr = np.array(arr)
        
        # 배열의 크기가 3 미만인 경우 예외 처리
        if len(arr) < 3:
            raise ValueError("Array must have at least 3 elements.")
        
        # 큰 값들의 인덱스를 구함
        sorted_indices = np.argsort(arr)
        top_3_indices = sorted_indices[::-1][:3]  # 내림차순으로 상위 3개 인덱스 선택
        
        # 큰 값들과 해당하는 인덱스를 반환
        top_3_values = arr[top_3_indices]
        top_3_indices = top_3_indices.tolist()
        
        return top_3_values, top_3_indices

    # 함수 호출하여 결과 얻기
    _, top_indices = get_top_3_values(probabilities_percent[0])

    commercial = pd.read_csv("commercial_with_xy_epsg4326_onehot.csv")

    import xgboost as xgb

    xgb_model = xgb.Booster()
    xgb_model.load_model('xgb_model_log.model')
    
    import math

    def get_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    df_coordinates = list(zip(df['lng'], df['lat']))
    commercial_coordinates = list(zip(commercial['new_x'], commercial['new_y']))

    min_dist = 100000000000000
    min_idx = 0

    for i in range(len(commercial)):
        dist = get_distance(df_coordinates[0], commercial_coordinates[i])
        if dist < min_dist:
            min_idx = i
            min_dist = dist

    commercial_info = commercial.iloc[min_idx]
    commercial_info = commercial_info.to_frame().T
    commercial_info = commercial_info.reset_index().drop(["index", "매출_log", "new_x", "new_y"], axis = 1)

    sectors = ['업종_명_가전제품_및_통신기기_도소매업',
        '업종_명_개인_서비스업', '업종_명_교육_및_사업자원_서비스업', '업종_명_기타_도소매업', '업종_명_기타_서비스업',
        '업종_명_부동산_및_임대업', '업종_명_생활용품_도소매업', '업종_명_수리업', '업종_명_숙박업',
        '업종_명_스포츠_및_오락_서비스업', '업종_명_식료품_도소매업', '업종_명_음식점_및_주점업', '업종_명_제조업', '업종_명_중고상품_도소매업'
        ]

    cnt = 1

    for i in top_indices:
        commercial_info.loc[:, sectors[i]] = 1.0
        sector = sectors[i]
        dtest = xgb.DMatrix(commercial_info)
        pred = xgb_model.predict(dtest)
        money = math.exp(pred[0]) / 100000000
        print(f"추천 업종 {cnt}위: {sector}\n예상 매출액: {money:>.1f}억원\n")
        cnt += 1
    
    return None

# 명령줄 인수로 전달된 JSON 데이터 추출
json_data = sys.argv[1]
json_data = pd.read_json(json_data)
# 계산 수행 및 결과 출력
calculation_result = perform_calculation(json_data)