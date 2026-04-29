# 목표

Vanilla TOSS로 사람 (연예인?) 사진 (상반신)에 대한 novel view를 뽑는 것이 목표인데, 기존 데이터셋인 objaverse로 잘 학습이 되어 있어서 fine-detail을 요구하지 않는 부분 (손, 옷 등)은 나름 잘 나오는 것 같다. 하지만 fine-detail을 요구하는 얼굴이 geometry를 잘 유지하지 못하고 fidelity가 높지 않은 것 같아 잘 나오는 부분은 최대한 유지한 채로 얼굴이 잘 나올 수 있도록 학습시키는 중이다.

# 제한 사항

- 완전한 3D views가 아닌 제한적인 (대략 -20~20도) views에 대해 novel views 생성

# 발생했던 문제들

- 2D → 3D Projection 후 3DGS 적용
    - Depth Map을 통해 2D이미지를 3D로 unproject하는 방식 활용
        - Occlusion
            - Ghosting Artifacts: 보이지 않는 부분에 대해 검은색 그림자가 만들어지는 점
                - 일부는 Noise를 활용해서 지우려고 했으나, 그러기에는 Artifact가 너무 큼
- Diffusion
    - Masking
    - Loss Oscillation

# 현재 아이디어

- LoRA
    - 효율적이되 얼굴 외 다른 부분은 최대한 건드리지 않는 방법을 고려하여 attention layer에 LoRA를 붙이고 이미지, geometry, conditioning 등을 활용해 학습시키고자 함

### 의문점

- LoRA가 얼굴의 Geometry를 학습시키기에 충분한가?
- 목표는 상반신의 Novel Views인데, 현재는 얼굴 데이터셋만 사용하고 있다. 얼굴 데이터셋만으로 상반신에서의 얼굴만 학습시킬 수 있을까?
    - 얼굴 인식해서 그부분만 고치도록 학습하는 방식을 고려해야 할까?
    - 마스킹 활용을 잘하면 될까?
- 배경은 어떻게 할까?

# 학습 과정

## 모델 구조

```python
[Diffusion UNet]
model.diffusion_model.base_model.model.time_embed # timestep embedding
model.diffusion_model.base_model.model.input_blocks # encoder/downsampler
model.diffusion_model.base_model.model.middle_blocks # bottleneck 의미/구조를 크게 조정
model.diffusion_model.base_model.model.output_blocks # decoder/upsampler 최종 디테일 복원
model.diffusion_model.base_model.model.out # noise prediction
model.diffusion_model.base_model.model.pose_net # camera pose conditioning

[VAE]
first_stage_model.encoder # encode to latent
first_stage_model.decoder # decode to image

[CLIP]
cond_stage_model.transformer.text_model.encoder # text embedding
```

1. **diffusion UNet**
    1. attention layer
        - `attn1`: **self-attention**
        - `attn2`: **cross-attention**
        - `ff.net`: feed-forward network
        - `norm1,2,3`: layer norm
2. **VAE (first stage model)**
3. **text encoder (CLIP 계열 cond stage model)**

### 추가 Layers/Unfrozen Layers

```python
Trainable parameters: 32
model.diffusion_model.base_model.model.middle_block.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight
model.diffusion_model.base_model.model.middle_block.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight
model.diffusion_model.base_model.model.middle_block.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight
model.diffusion_model.base_model.model.middle_block.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight
model.diffusion_model.base_model.model.middle_block.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight
model.diffusion_model.base_model.model.middle_block.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.9.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.9.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.9.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.9.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.10.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.10.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.10.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.10.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.11.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.11.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight
model.diffusion_model.base_model.model.output_blocks.11.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight
model.diffusion_model.base_model.model.output_blocks.11.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight
model.diffusion_model.base_model.model.out.0.weight
model.diffusion_model.base_model.model.out.0.bias
model.diffusion_model.base_model.model.out.2.weight
model.diffusion_model.base_model.model.out.2.bias
model.diffusion_model.base_model.model.pose_net.0.weight
model.diffusion_model.base_model.model.pose_net.0.bias
model.diffusion_model.base_model.model.pose_net.2.weight
model.diffusion_model.base_model.model.pose_net.2.bias

```

UNet 구조에서 input_blocks, middle_blocks, output_blocks로 나눴을 때

- input_blocks는 encoder/down path라서 처음에는 아직 feature가 **입력 이미지/latent의 로컬 패턴** 쪽에 더 가까움
- middle_blocks는 해상도가 가장 낮고 receptive field가 가장 넓은 구간이라 **전역적인 구조와 관계**

를 다루기 좋음

- output_blocks는 decoder/up path라서 중간에서 정한 의미/구조를 바탕으로 **세부 복원.** 즉, detail을 다루는 부분임

그래서, LoRA를 TOSS 네트워크 중에서 얼굴에 중요한 구조와 디테일에 초점을 맞추기 위해서 middle_blocks와 output_blocks (후반부)의 self-attention layer에다가 LoRA를 붙였다.

## Dataset

### 사용한 모듈

- pixel3dmm
- celebA
- portrait4d

### Features

- RGB
- Depth
- Normal
- Mask
- Pose

## Dataset 준비
### Synthetic Dataset 생성
Portrait4d를 활용하여 CelebA Dataset 중 100장의 이미지를 골라 Multiview를 생성


### Dataset의 Background 없애기
TOSS는 배경이 있으면 결과물이 이상하게 나오기 때문에, 배경없이 학습하는 게 좋다. 그래서, 배경을 미리 모든 CelebA 이미지에서 없애준다.


### Dataset에 Normal Map 포함시키기
pixel3dmm이 얼굴에 대한 novel views를 충분히 잘 뽑는다고 생각하여 celebA의 novel views를 갖고 LoRA를 학습시킴

- Ground Truth 생성용 스크립트 작성
  - input/에 있는 각 dataset에 대해 preprocessing 후 각 identity 폴더에 분배
- Dataset normals load 하도록 수정
- 학습 루틴에 넣기
    - Proxy estimator 사용
      - DPT와 같은 비교적 가벼운 geometry 모델을 teacher model로 사용하여 inference된 결과를 사용한 Loss를 통해 학습
    - Crop된 normal uncrop
      pixel3dmm의 결과는 crop된(얼굴 확대된) 결과라서 원본 이미지와 얼굴/배경 비중이 달라서 pixel3dmm에서 사용했던 bounding box를 통해 따로 align 시켜야 함
        - 방법1: RGB를 학습할 때 똑같이 crop한다
        - 방법2: 추출된 Normal를 다시 uncrop한 다음 face mask를 적용
            - pixel3dmm에서 crop했을 때의 정보를 통해 normals를 uncrop한 뒤, TOSS_LORA에 사용했던 mask를 적용한 뒤 loss 적용

### Dataset에 Depth Map 포함시키기
- 

### Dataset Load

- Normal Mask
  - Valid한 부분만 필터링

- Depth Mask
  - Valid한 부분만 필터링


# 현재 상황
## Vanilla TOSS의 한계
- 전신 사진은 나름 뽑음
- 얼굴은 잘 못 뽑는다
  - 얼굴이 얼굴이 아니라 종이쪼가리로 보는 것 같음
- 투명배경의 이미지여야 한다

## 학습이 안 됨
총 500 steps 정도 돌리는데, MSE랑 Perceptual 등 Loss가 Converge하지 않고 Oscillate

### 가능성
1. PoseNet이랑 Out을 얼려서?
  - Unfreeze시켰던 PoseNet과 Out 레이어를 Freeze시킨 결과, 똑같음
2. Geometry 정보가 없어서?
  - Depth, Normal 추가했지만 똑같음
3. LoRA를 더 많은 레이어에 적용해야 하나?
  - 현재는 self-attention layer에만 LoRA를 추가한 상태. Pose의 Conditioning이 잘 학습되려면 cross-attention layer에도 LoRA를 추가해야 하는 것이 아닌지 의문
    - PoseNet은 그럼 unfreeze 시켜야 하는건가?
4. Learning Rate가 너무 높나?
5. Masking이 잘 적용이 안 됐나?

![image.png](attachment:82bea87c-29fd-4210-8fdf-f5260c7dec84:image.png)

![image.png](attachment:26571fae-1894-4f63-be03-0a865f61d9da:image.png)

Gradient Norm


# 참고
- humanNorm
    - how it handles geometry
- depth head architecture
    - how should the architecture be
- proxy estimator
    - how it's different from applying depth head
- background removal
    - is it applied properly or is it still using background?
- adding upper-body dataset
  - 얼굴에 대해서 Portrait4d 데이터셋 생성
  - ex)
      - Celeb-FBI
      - DeepFashion
- inference 시 face-mask 적용
- Conditioning