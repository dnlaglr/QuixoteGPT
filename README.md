# QuixoteGPT

## Data and Statistics

**Difference in data from 1st iteration to 5,000th:**  
Total Training Time: **3.8 Hours**  
Training Loss: **4.4665 &rarr; 1.0845**  
Validation Loss: **4.4678 &rarr; 1.1984**  

![Training and Validation Loss](/assets/loss-graph.svg)

Total Training Time: **3.8 Hours**  
Training Loss: **4.4665 &rarr; 1.0845**  
Validation Loss: **4.4678 &rarr; 1.1984**  

## Training and Generation

See the completed 10,000 token generation at **[`docs/generatedText.md`](docs/generatedText.md "Navigate to markdown")**.

See the generation progression during training at **[`docs/genProgress.md`](docs/genprogress.md "Navigate to markdown")**.

## Limitations

- Hardware was a massive limiting factor for training this model. Using a Nvidia GeForce 1060 3gb graphics card alongside CUDA took 4 hours to finish training and was at practically 100% usage the entire time.