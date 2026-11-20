#include <string.h>
static float layer_checkpoints[5][64];
void capture_layer(int l,float*x,int d){memcpy(layer_checkpoints[l],x,d*4);}
#define main original_main
#include "golden_run.c"
#undef main
static FILE *output(const char *path){FILE*f=fopen(path,"wb");if(!f){perror(path);exit(1);}return f;}
int main(){
 Transformer t;build_transformer(&t,"dist/fixtures/weights.bin");
 FILE*all=output("validation/all_logits.bin");int kd=t.config.dim*t.config.n_kv_heads/t.config.n_heads,stride=t.config.seq_len*kd;
 for(int pos=0;pos<t.config.seq_len;pos++){
  int fixture=pos==0||pos==127||pos==511;char name[100];
  if(fixture){sprintf(name,"dist/fixtures/prefix_%d.bin",pos+1);FILE*f=output(name);for(int l=0;l<t.config.n_layers;l++){fwrite(t.state.key_cache+l*stride,4,pos*kd,f);fwrite(t.state.value_cache+l*stride,4,pos*kd,f);}fclose(f);}
  float*r=forward(&t,1,pos);fwrite(r,4,t.config.vocab_size,all);
  if(fixture){sprintf(name,"dist/fixtures/golden_%d.bin",pos+1);FILE*f=output(name);fwrite(r,4,t.config.vocab_size,f);for(int l=0;l<t.config.n_layers;l++){fwrite(t.state.key_cache+l*stride,4,(pos+1)*kd,f);fwrite(t.state.value_cache+l*stride,4,(pos+1)*kd,f);}fwrite(layer_checkpoints,4,5*64,f);fclose(f);}
 }
 fclose(all);FILE*f=output("validation/final_kv.bin");for(int l=0;l<t.config.n_layers;l++){fwrite(t.state.key_cache+l*stride,4,stride,f);fwrite(t.state.value_cache+l*stride,4,stride,f);}fclose(f);free_transformer(&t);return 0;
}
