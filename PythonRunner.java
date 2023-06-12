import java.io.*;

// [1]
// 암호 손동작 설정. <open setMotion.py>
// docName, action 전달 -> true/false 여부 리턴

// action을 first, second, third로 바꿔가면서 총 3번 setMotion.py 열어야 함

// docName, action -> 손동작 데이터 파일명에 사용됨
// 손동작 정보는 dataset 폴더에 seq_docName_action.npy저장
// seq 데이터 부족한 경우 다시 수행해야함. 손을 아에 인식 못하면 (0,) 이런식으로 뜸.
// 시퀀스 개수 300이하일 경우 충분하지 않다고 판단, false 전달 하도록 했음. 충분할 경우 true


public class PythonRunner {
    public static void main(String[] args) throws IOException, InterruptedException {
        // parameters : document name, action
        // action을 first, second, third 이렇게 총 3번 파이썬 파일 열어야 함!
        String docName = "final2";
        String action = "third";

        // activate anaconda virtual environment, execute python file
        ProcessBuilder pb = new ProcessBuilder();
        pb.directory(new File("/Users/seungtoc/Desktop/motion"));
        pb.command("/Users/seungtoc/anaconda3/bin/conda", "run", "-n", "motion", "python", "setMotion.py", docName, action);
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
            System.out.println("setMotionFor.py executed successfully.");
        } else {
            System.out.println("setMotionFor.py execution failed.");
        }
    }
}
