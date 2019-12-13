#include <kipr/botball.h>

int main() {
  // Initialize variables
  int lightReading = 0;

  while(1) {
    // Get light reading from analog sensor
    lightReading = analog(1);
    // Display light reading
    printf("The current light reading is %d\n", lightReading);

    // Wait 0.5s
    wait_for_milliseconds(500);
  }

  return 0;
}
