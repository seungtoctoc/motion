import java.io.*;

// [2] 
// 모델 학습. open trainMoiton.py
// docName 전달 -> 리턴값 없음

// docName 이용해 dataset 폴더에서 손동작 시퀀스 데이터 찾고, 학습, 모델 생성
// 학습한 모델은 models 폴더에 model_{docName}.h5 로 저장될 것
// 학습 과정 2~3분 소요
// 정상적으로 수행하고 종료되면 아래 exitCode==0 조건문 실행되는데 그거 이용해서 서버 진행하면 될듯

public class PythonRunner2 {
    public static void main(String[] args) throws IOException, InterruptedException {
        // parameters : document name
        // 학습에서 손동작 데이터를 정해진 위치에 저장하고 학습에도 그 경로 이용해서 문서 이름만 파라미터로 전달하면 됨!
        String docName = "test9020";

        // activate anaconda virtual environment, execute python file
        ProcessBuilder pb = new ProcessBuilder();
        pb.command("/Users/seungtoc/anaconda3/bin/conda", "run", "-n", "motion", "python", "trainMotion.py", docName);
        Process process = pb.start();


        // get outputs
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }


        //  for confirmation
        int exitCode = process.waitFor();
        if (exitCode == 0) {
            System.out.println("trainMotion.py executed successfully.");
        } else {
            System.out.println("trainMotion.py execution failed.");
        }
    }
}
