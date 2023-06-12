import java.io.*;

// [3] 
// 손동작 암호 해제 <testMotion.py>
// docName, action 전달 -> true/false 여부 리턴

// action을 first, second, third로 바꿔가면서 총 3번 testMotion.py 열어야함
// docName으로 모델 찾고, 해당 모델로 동작 판단
// 파라미터로 받은 액션 수행 시 true 반환. 30초 안에 액션 수행 안했다고 판단하면 false 반환

public class PythonRunner3 {
    public static void main(String[] args) throws IOException, InterruptedException {
        // parameters : document name
        String docName = "final";
        String action = "third";

        // activate anaconda virtual environment, execute python file
        ProcessBuilder pb = new ProcessBuilder();
        pb.command("/Users/seungtoc/anaconda3/bin/conda", "run", "-n", "motion", "python", "testMotion.py", docName, action);
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
            System.out.println("testMotion.py executed successfully.");
        } else {
            System.out.println("testMotion.py execution failed.");
        }
    }
}
