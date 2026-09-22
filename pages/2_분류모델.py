import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("🤖 뇌졸중 분류 모델 만들기")
st.markdown("나이, 혈당 등의 정보를 이용해서 뇌졸중 여부를 예측하는 모델을 만들어봅니다.")
st.markdown("여기서는 **stroke = 1(뇌졸중 있음)을 양성**으로 둡니다.")

st.divider()

# -----------------------------
# 열 이름 <-> 우리말 이름 매핑
# -----------------------------
col_kor_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
kor_col_map = {v: k for k, v in col_kor_map.items()}  # 우리말 -> 원래 열 이름

all_features_kor = list(col_kor_map.values())
default_features_kor = [v for k, v in col_kor_map.items() if k != "bmi"]  # bmi 제외 기본값

# -----------------------------
# 1. 입력 속성 선택
# -----------------------------
st.subheader("1️⃣ 입력으로 사용할 속성 고르기")

selected_kor = st.multiselect(
    "모델의 입력으로 사용할 속성을 골라주세요 (처음에는 체질량지수를 뺀 4개가 선택되어 있습니다).",
    options=all_features_kor,
    default=default_features_kor
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 2개 이상 골라주세요. 현재는 모델을 만들 수 없습니다.")
    st.stop()

# 우리말 선택 -> 원래 영문 열 이름으로 변환
selected_cols = [kor_col_map[k] for k in selected_kor]
use_bmi = "bmi" in selected_cols

st.success(f"선택한 속성: {', '.join(selected_kor)}")

st.divider()

# -----------------------------
# 2. 데이터 준비: id 순 정렬 후 10명씩 묶어 앞 3명 테스트용
# -----------------------------
st.subheader("2️⃣ 학습용 / 테스트용 데이터 나누기")

data = df.copy()
data = data.sort_values("id").reset_index(drop=True)

# 10명씩 묶었을 때, 그룹 안에서의 순서(0~9)를 계산
data["순서"] = np.arange(len(data)) % 10

# 순서 0,1,2 -> 테스트용 / 순서 3~9 -> 학습용
test_mask = data["순서"] < 3
train_mask = ~test_mask

train_df = data[train_mask].copy()
test_df = data[test_mask].copy()

st.markdown(f"- 전체 사람 수: **{len(data):,}명**")
st.markdown(f"- 학습용 사람 수: **{len(train_df):,}명**")
st.markdown(f"- 테스트용 사람 수: **{len(test_df):,}명**")

# -----------------------------
# 3. bmi 결측치 처리 (학습용 중앙값으로 채우기) - bmi를 선택한 경우에만
# -----------------------------
if use_bmi:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.info(f"💡 체질량지수(bmi)의 빈 값은 학습용 데이터의 중앙값인 **{bmi_median:.2f}**로 채웠습니다.")

X_train = train_df[selected_cols]
y_train = train_df["stroke"]
X_test = test_df[selected_cols]
y_test = test_df["stroke"]

st.divider()

# -----------------------------
# 4. 모델 학습
# -----------------------------
st.subheader("3️⃣ 모델 학습하기")

# 로지스틱 회귀 (확률로 답하는 모델)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# 의사결정트리 (질문으로 답하는 모델) - 깊이 3, 최소 5명 미만이면 그만 나누기, 난수 고정
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

# 더미 모델 (입력을 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# 각 모델의 훈련/테스트 정확도 계산
def get_accuracies(model):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model)
tree_train_acc, tree_test_acc = get_accuracies(tree_model)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model)

st.markdown("모델 학습이 완료되었습니다!")

st.divider()

# -----------------------------
# 5. 정확도 카드
# -----------------------------
st.subheader("4️⃣ 모델별 정확도 비교")

card1, card2, card3 = st.columns(3)

with card1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{log_test_acc*100:.2f} %"
    )
    st.caption(f"훈련 정확도: {log_train_acc*100:.2f}% ・ 테스트 정확도: {log_test_acc*100:.2f}%")

with card2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{tree_test_acc*100:.2f} %"
    )
    st.caption(f"훈련 정확도: {tree_train_acc*100:.2f}% ・ 테스트 정확도: {tree_test_acc*100:.2f}%")

with card3:
    st.metric(
        label="기준 모델(항상 많은 쪽으로만 답하는 모델)",
        value=f"{dummy_test_acc*100:.2f} %"
    )
    st.caption(f"훈련 정확도: {dummy_train_acc*100:.2f}% ・ 테스트 정확도: {dummy_test_acc*100:.2f}%")

st.divider()

# -----------------------------
# 6. 산점도 + 로지스틱 회귀 결정 경계 + 트리 영역 색칠
# -----------------------------
st.subheader("5️⃣ 산점도로 살펴보기")

if len(selected_kor) < 2:
    st.warning("산점도를 그리려면 속성이 2개 이상 필요합니다.")
    st.stop()

col_x_kor, col_y_kor = st.columns(2)
with col_x_kor:
    x_kor = st.selectbox("가로축으로 사용할 속성", options=selected_kor, index=0)
with col_y_kor:
    remaining = [k for k in selected_kor if k != x_kor]
    y_kor = st.selectbox("세로축으로 사용할 속성", options=remaining, index=0)

x_col = kor_col_map[x_kor]
y_col = kor_col_map[y_kor]

# 축으로 쓰지 않는 나머지 속성들은 테스트 데이터의 중앙값으로 고정
other_cols = [c for c in selected_cols if c not in [x_col, y_col]]
fixed_values = {c: test_df[c].median() for c in other_cols}

if fixed_values:
    fixed_text = ", ".join([f"{col_kor_map[c]} = {v:.2f}" for c, v in fixed_values.items()])
    st.markdown(f"📌 그림에 나타나지 않는 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.markdown("📌 선택한 속성이 축 2개뿐이라 고정할 다른 속성이 없습니다.")

# 격자(그리드) 생성: x, y 값을 촘촘하게 만들어 배경 영역 색칠에 사용
x_min, x_max = test_df[x_col].min(), test_df[x_col].max()
y_min, y_max = test_df[y_col].min(), test_df[y_col].max()

x_range = np.linspace(x_min, x_max, 200)
y_range = np.linspace(y_min, y_max, 200)
xx, yy = np.meshgrid(x_range, y_range)

grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for c, v in fixed_values.items():
    grid_df[c] = v

# 모델이 학습할 때 사용한 컬럼 순서에 맞추기
grid_df = grid_df[selected_cols]

# 의사결정트리 예측 -> 배경 영역 색칠용
tree_grid_pred = tree_model.predict(grid_df).reshape(xx.shape)

fig = go.Figure()

# 트리의 배경 영역 (옅은 색)
fig.add_trace(go.Contour(
    x=x_range, y=y_range, z=tree_grid_pred,
    showscale=False,
    colorscale=[[0, "rgba(99,110,250,0.15)"], [1, "rgba(239,85,59,0.15)"]],
    contours=dict(start=0, end=1, size=1),
    line_width=0,
    hoverinfo="skip",
    name="의사결정트리 영역"
))

# 테스트 데이터 산점도 (실제 뇌졸중 여부로 색 구분)
test_df_plot = test_df.copy()
test_df_plot["실제_뇌졸중"] = test_df_plot["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

for label, color in [("뇌졸중 없음", "#636EFA"), ("뇌졸중 있음", "#EF553B")]:
    subset = test_df_plot[test_df_plot["실제_뇌졸중"] == label]
    fig.add_trace(go.Scatter(
        x=subset[x_col], y=subset[y_col],
        mode="markers",
        marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
        name=label
    ))

# 로지스틱 회귀의 0.5 결정 경계선 계산
# 로지스틱 회귀: w0*x0 + w1*x1 + ... + b = 0 일 때 확률이 0.5
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]

col_index = {c: i for i, c in enumerate(selected_cols)}
w_x = coef[col_index[x_col]]
w_y = coef[col_index[y_col]]

# 고정된 다른 속성들의 영향을 절편에 더해줌
fixed_contribution = sum(coef[col_index[c]] * v for c, v in fixed_values.items())
b_total = intercept + fixed_contribution

line_drawn = False
if abs(w_y) > 1e-12:
    # y = -(w_x*x + b_total) / w_y
    y_line = -(w_x * x_range + b_total) / w_y
    # 그림 범위 안에 선이 어느 정도 걸치는지 확인
    in_range_mask = (y_line >= y_min) & (y_line <= y_max)
    if in_range_mask.any():
        fig.add_trace(go.Scatter(
            x=x_range, y=y_line,
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name="로지스틱 회귀 경계선(0.5)"
        ))
        line_drawn = True
elif abs(w_x) > 1e-12:
    # w_y가 0이면 x = -b_total / w_x 인 수직선
    x_line_val = -b_total / w_x
    if x_min <= x_line_val <= x_max:
        fig.add_trace(go.Scatter(
            x=[x_line_val, x_line_val], y=[y_min, y_max],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name="로지스틱 회귀 경계선(0.5)"
        ))
        line_drawn = True

fig.update_layout(
    xaxis_title=x_kor,
    yaxis_title=y_kor,
    title="테스트 데이터 산점도와 로지스틱 회귀 경계선, 의사결정트리 영역",
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(fig, use_container_width=True)

if not line_drawn:
    st.info("📌 로지스틱 회귀의 0.5 경계선은 이 그림의 범위 밖에 있어서 표시되지 않았습니다.")

st.divider()

# -----------------------------
# 7. 의사결정트리 가지 그림 (graphviz DOT)
# -----------------------------
st.subheader("6️⃣ 의사결정트리가 던진 질문 살펴보기")

tree_ = tree_model.tree_
feature_names = selected_cols

def build_dot(tree_, feature_names, kor_map):
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled", fontname="Malgun Gothic"];']

    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value  # 각 노드의 클래스별 샘플 수 [ [음성수, 양성수] ]

    for i in range(n_nodes):
        n_samples = int(value[i][0][0] + value[i][0][1])
        n_positive = int(value[i][0][1])
        ratio = n_positive / n_samples if n_samples > 0 else 0

        is_leaf = children_left[i] == children_right[i]

        if is_leaf:
            # 답을 내는 마디(리프): 다수결로 답 결정
            predicted_label = "뇌졸중 있음" if n_positive > (n_samples - n_positive) else "뇌졸중 없음"
            color = "#EF553B" if predicted_label == "뇌졸중 있음" else "#636EFA"
            label = f"인원: {n_samples}명\\n뇌졸중: {n_positive}명\\n비율: {ratio*100:.1f}%\\n답: {predicted_label}"
            dot_lines.append(f'{i} [label="{label}", fillcolor="{color}", fontcolor="white"];')
        else:
            # 질문을 하는 마디
            f_name_kor = kor_map[feature_names[feature[i]]]
            th = threshold[i]
            label = f"{f_name_kor} <= {th:.2f} ?\\n인원: {n_samples}명\\n뇌졸중: {n_positive}명\\n비율: {ratio*100:.1f}%"
            dot_lines.append(f'{i} [label="{label}", fillcolor="#F0F2F6", fontcolor="black"];')

    for i in range(n_nodes):
        left = children_left[i]
        right = children_right[i]
        if left != -1:
            dot_lines.append(f'{i} -> {left} [label="예"];')
        if right != -1:
            dot_lines.append(f'{i} -> {right} [label="아니요"];')

    dot_lines.append("}")
    return "\n".join(dot_lines)

dot_str = build_dot(tree_, feature_names, col_kor_map)
st.graphviz_chart(dot_str)

st.divider()

# -----------------------------
# 8. 트리 요약 정보
# -----------------------------
st.subheader("7️⃣ 트리 요약")

# 리프 노드(답을 내는 마디) 정보 추출
n_nodes = tree_.node_count
children_left = tree_.children_left
children_right = tree_.children_right
feature = tree_.feature
value = tree_.value

leaf_count = 0
negative_leaf_count = 0
used_features = set()

for i in range(n_nodes):
    is_leaf = children_left[i] == children_right[i]
    if is_leaf:
        leaf_count += 1
        n_samples = value[i][0][0] + value[i][0][1]
        n_positive = value[i][0][1]
        n_negative = value[i][0][0]
        if n_negative > n_positive:
            negative_leaf_count += 1
    else:
        used_features.add(feature_names[feature[i]])

st.markdown(f"- 답을 내는 마디(리프 노드)는 모두 **{leaf_count}칸**이고, 그중 **{negative_leaf_count}칸**이 '뇌졸중 없음'이라고 답합니다.")

if used_features:
    used_features_kor = [col_kor_map[c] for c in used_features]
    st.markdown(f"- 선택한 속성 가운데 이 나무가 실제로 질문에 사용한 속성: **{', '.join(used_features_kor)}**")
else:
    st.markdown("- 이 나무는 어떤 속성으로도 질문을 만들지 않았습니다 (모든 사람이 같은 답을 받습니다).")

not_used_features = [col_kor_map[c] for c in selected_cols if c not in used_features]
if not_used_features:
    st.markdown(f"- 선택했지만 이 나무가 사용하지 않은 속성: **{', '.join(not_used_features)}**")
