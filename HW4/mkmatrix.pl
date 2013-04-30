#!/usr/bin/perl -w
#

die "Usage: $0 <matrix-size>\n" if ($#ARGV!=0);
$n = shift(@ARGV);
srand($n);


#=============================================================================
# This function randomly permutes the input vector and returns it
#=============================================================================
sub RandomPermuteVector($) 
{
  my @order = @{$_[0]};
  my ($n, $i, $j, $k);

  $n = $#order+1;
  for ($k=0; $k<20; $k++) {
    for ($i=0; $i<$n; $i++) {
      $j = int(rand($n));
      $tmp = $order[$i];
      $order[$i] = $order[$j];
      $order[$j] = $tmp;
    }
  }
  return @order;
}
#=============================================================================




# Create the random permutation
for ($i=0; $i<$n; $i++) {
  $perm[$i] = $i;
}
@perm = RandomPermuteVector(\@perm);


# Create the non-zero elements of the matrix
$k = 0;
for ($i=0; $i<$n; $i++) {
  $w = 25 + int(rand(25));
  $w = ($w < sqrt($n) ? $w : int(sqrt($n)));
  for ($j=$i-3*$w; $j<=$i+3*$w; $j+=3) {
    if ($j >= 0 && $j<$n) {
      $A[$k][0] = $i;
      $A[$k][1] = $j;
      $A[$k][2] = rand(5)*(rand(6) > 3 ? +1 : -1);
      $k++;
    }
  }
  $b[$i] = rand(5)*(rand(6) > 3 ? +1 : -1);
}

# Write the first matrix 
$filename = "m$n-A.ij";
open(FPOUT, ">$filename");
#print FPOUT "$n\n";
for ($i=0; $i<$k; $i++) {
  print FPOUT $A[$i][0], " ", $A[$i][1], " ", $A[$i][2], "\n";
}
close(FPOUT);

# Write the second matrix 
$filename = "m$n-C.ij";
open(FPOUT, ">$filename");
#print FPOUT "$n\n";
for ($i=0; $i<$k; $i++) {
  print FPOUT $perm[$A[$i][0]], " ", $perm[$A[$i][1]], " ", $A[$i][2], "\n";
}
close(FPOUT);

# Sort the second matrix
$filename2 = "m$n-B.ij";
system("cat $filename | sort -g -k 2,2 | sort -g -k 1,1 -s > $filename2");
unlink($filename);

# Write the b vector
$filename = "m$n.vec";
open(FPOUT, ">$filename");
for ($i=0; $i<$n; $i++) {
  print FPOUT $b[$i], "\n";
}
close(FPOUT);
